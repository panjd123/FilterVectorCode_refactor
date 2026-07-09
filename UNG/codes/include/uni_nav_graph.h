#ifndef UNG_H
#define UNG_H
#include "trie.h"
#include "graph.h"
#include "storage.h"
#include "distance.h"
#include "search_cache.h"
#include "label_nav_graph.h"
#include "ung_build_config.h"
#include "ung_cross_edge_config.h"
#include "ung_cross_edge_result.h"
#include "ung_query_stats.h"
#include "ung_acorn_augment_config.h"
#include "ung_group_graph_build_types.h"
#include "ung_query_route.h"
#include "ung_entry_group.h"
#include "ung_graph_search_backend.h"
#include "ung_cross_edge_output_writer.h"
#include "ung_special_blocks.h"
#include <unordered_map>
#include <bitset>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <boost/dynamic_bitset.hpp>
#include <roaring/roaring.h>
#include <roaring/roaring.hh>


using BitsetType = boost::dynamic_bitset<>;

class MethodSelector;

namespace faiss
{
   struct IndexACORNFlat;
}

namespace ANNS
{
   class Vamana;
   class GpuCoverFrontierProvider;

   class UniNavGraph
   {
   public:
      UniNavGraph(IdxType num_nodes);
      UniNavGraph();
      ~UniNavGraph();

      // Primary index lifecycle API.
      void build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                 std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                 IdxType max_degree, IdxType Lbuild, float alpha, std::string dataset,
                 ANNS::AcornInUng new_cross_edge);

      void calculate_query_features_only(
          std::shared_ptr<IStorage> query_storage,
          uint32_t num_threads,
          const std::string &output_csv_path,
          bool is_new_trie_method,
          bool is_rec_more_start);
      void search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                 uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                 IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                 std::vector<std::bitset<10000001>> &bitmap);
      void search_hybrid(std::shared_ptr<IStorage> &query_storage,
                         std::shared_ptr<DistanceHandler> &distance_handler,
                         const SearchRuntimeConfig &runtime,
                         std::pair<IdxType, float> *results,
                         std::vector<float> &num_cmps,
                         std::vector<QueryStats> &query_stats,
                         const std::vector<IdxType> &true_query_group_ids = {});
      void search_hybrid(std::shared_ptr<IStorage> &query_storage,
                         std::shared_ptr<DistanceHandler> &distance_handler,
                         uint32_t num_threads, IdxType Lsearch,
                         IdxType num_entry_points, std::string scenario,
                         IdxType K, std::pair<IdxType, float> *results,
                         std::vector<float> &num_cmps,
                         std::vector<QueryStats> &query_stats,
                         bool is_ori_ung,
                         bool is_select_entry_groups, bool is_rec_more_start,
                         bool is_ung_more_entry,
                         int lsearch_start, int lsearch_step,
                         int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold, 
                         int force_use_alg, bool is_bfs_filter,
                         const std::vector<IdxType> &true_query_group_ids = {});

      // I/O
      void save(std::string index_path_prefix, std::string results_path_prefix);
      void load(std::string index_path_prefix, std::string selector_model_prefix, const std::string &data_type, const std::string &acorn_index_path, const std::string &acorn_1_index_path, const std::string &dataset);
      void load_bipartite_graph(const std::string &filename);

      // Diagnostic and benchmark-facing helpers. These are intentionally
      // public because search_UNG_index and bitmap benchmarks call them
      // directly; new production search providers should go through
      // prepare_entry_groups_for_execution() instead of expanding this API.
      static bool compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2);
      std::pair<std::bitset<10000001>, double> compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const;
      roaring::Roaring compute_bitmap_from_groups(const std::vector<IdxType> &group_ids) const;
      std::vector<roaring::Roaring> batch_compute_ung_bitmaps(
          const ANNS::UniNavGraph &index,
          const std::shared_ptr<ANNS::IStorage> &query_storage,
          uint32_t num_threads,
          bool is_new_trie_method,
          bool is_rec_more_start);

      std::vector<IdxType> select_entry_groups(
          const std::vector<IdxType> &minimum_entry_sets,
          SelectionMode mode,
          size_t top_k,
          double beta = 1.0,
          IdxType true_query_group_id = 0) const;

      void get_min_super_sets_debug(const std::vector<LabelType> &query_label_set,
                                    std::vector<IdxType> &min_super_set_ids,
                                    bool avoid_self, bool need_containment,
                                    std::atomic<int> &print_counter, bool is_new_trie_method, bool is_rec_more_start, QueryStats &stats,
                                    bool skip_group_id_check) const;

      void warmup_selectors(uint32_t num_threads);

   private:

      // Search runtime orchestration. This layer owns query-level threading,
      // runtime config fan-out, and result writeback only.
      // Implementation: uni_nav_graph_search.cpp.
      void thread_function(IdxType query_id,
                           const SearchRuntimeConfig &runtime,
                           const GraphSearchBackend &graph_backend,
                           std::pair<IdxType, float> *results,
                           std::vector<float> &num_cmps,
                           std::vector<QueryStats> &query_stats,
                           const std::vector<IdxType> &true_query_group_ids);

      // Query route policy. It decides which high-level route a query should
      // take, but does not materialize final entry groups or execute graph
      // expansion.
      // Implementation: uni_nav_graph_query_route.cpp.
      QueryRouteDecision decide_query_route(const std::vector<LabelType> &query_labels,
                                            bool is_idea2_available,
                                            bool is_new_trie_method,
                                            bool is_rec_more_start,
                                            int force_use_alg,
                                            bool is_bfs_filter,
                                            std::vector<IdxType> &entry_group_ids,
                                            QueryStats &stats);

      // Entry-group provider. It is the boundary for CPU/GPU/minimal/non-
      // minimal entry-group implementations; search backends consume only the
      // resulting group IDs.
      // Implementation: uni_nav_graph_entry_provider.cpp.
      void prepare_entry_groups_for_execution(const EntryGroupProviderRequest &request,
                                              std::vector<IdxType> &entry_group_ids,
                                              QueryStats &stats);
      EntryGroupProviderResult run_entry_group_provider(const EntryGroupProviderRequest &request,
                                                        QueryStats &stats);
      EntryGroupProviderResult compute_cpu_entry_groups_for_execution(const EntryGroupProviderRequest &request,
                                                                      QueryStats &stats);
      EntryGroupProviderResult compute_gpu_entry_groups_for_execution(const EntryGroupProviderRequest &request,
                                                                      QueryStats &stats);

      // Search backends. These execute ACORN-compatible or UNG graph expansion
      // after the route and entry-group provider have finished.
      // Implementation: uni_nav_graph_search_backend.cpp.
      bool execute_acorn_query(const char *query,
                               const std::vector<LabelType> &query_labels,
                               const std::vector<IdxType> &entry_group_ids,
                               IdxType query_id,
                               IdxType K,
                               IdxType Lsearch,
                               int lsearch_start,
                               int lsearch_step,
                               int efs_start,
                               int efs_step_slow,
                               int efs_step_fast,
                               int lsearch_threshold,
                               int force_use_alg,
                               int algorithm,
                               bool is_bfs_filter,
                               bool use_old_bitmap_search,
                               std::pair<IdxType, float> *results,
                               std::vector<float> &num_cmps,
                               SearchQueue &cur_result,
                               QueryStats &stats);
      bool execute_ung_query(const char *query,
                             std::shared_ptr<SearchCache> search_cache,
                             const GraphSearchBackend &graph_backend,
                             const std::vector<IdxType> &entry_group_ids,
                             IdxType query_id,
                             IdxType num_entry_points,
                             std::vector<float> &num_cmps,
                             SearchQueue &cur_result,
                             QueryStats &stats);
      bool execute_special_block_ung_query(const char *query,
                                           const SearchRuntimeConfig &runtime,
                                           const GraphSearchBackend &graph_backend,
                                           const std::vector<IdxType> &entry_group_ids,
                                           const std::vector<LabelType> &query_labels,
                                           IdxType query_id,
                                           std::vector<float> &num_cmps,
                                           SearchQueue &cur_result,
                                           QueryStats &stats);

      // Query feature and selector diagnostics. These helpers feed route
      // decisions and benchmark CSVs; they should stay side-effect-light.
      // Implementation: uni_nav_graph_query_features.cpp.
      size_t get_candidate_count_for_label(LabelType label) const;
      std::vector<float> calculate_idea1_features(const QueryStats &stats) const;
      std::vector<float> calculate_idea2_features(const QueryStats &stats) const;
      std::optional<bool> check_pre_trie_heuristic(const std::string& dataset_name, size_t query_length, size_t candidate_set_size) const;
      void populate_entry_group_route_stats(const std::vector<IdxType> &entry_group_ids,
                                            QueryStats &stats) const;

      // Low-level UNG graph expansion helpers used by the search backend and
      // by legacy cross-edge construction paths.
      // Implementation: uni_nav_graph_search_backend.cpp.
      std::vector<IdxType> get_entry_points(const std::vector<LabelType> &query_label_set,
                                            IdxType num_entry_points, VisitedSet &visited_set);
      void get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                           IdxType group_id, std::vector<IdxType> &entry_points);
      IdxType iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                     IdxType target_id, const std::vector<IdxType> &entry_points,
                                     size_t &num_nodes_visited,
                                     bool clear_search_queue = true, bool clear_visited_set = true);
      IdxType iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                     const GraphSearchBackend &graph_backend,
                                     IdxType target_id, const std::vector<IdxType> &entry_points,
                                     size_t &num_nodes_visited,
                                     bool clear_search_queue = true, bool clear_visited_set = true);
      CrossEdgeCsrOutput build_search_graph_csr() const;

      // data
      std::shared_ptr<IStorage> _base_storage,
          _query_storage;
      std::shared_ptr<DistanceHandler> _distance_handler;
      std::shared_ptr<Graph> _graph;
      IdxType _num_points = 0;

      // Vector-attribute bipartite graph. This supports bitmap generation,
      // persistence, and ACORN/filter compatibility; it is not a backend
      // extension point.
      std::vector<std::vector<IdxType>> _vector_attr_graph;
      std::unordered_map<LabelType, AtrType> _attr_to_id;
      std::unordered_map<AtrType, LabelType> _id_to_attr;
      AtrType _num_attributes = 0;

      // trie index and vector groups
      IdxType _num_groups = 0;
      TrieIndex _trie_index;
      std::vector<IdxType> _new_vec_id_to_group_id;
      std::vector<std::vector<IdxType>> _group_id_to_vec_ids;
      std::vector<std::vector<LabelType>> _group_id_to_label_set;
      void build_trie_and_divide_groups();

      // Special block overlay. Disabled by default; enabled with
      // UNG_SPECIAL_BLOCKS=1. Construction lives in
      // uni_nav_graph_special_blocks.cpp.
      std::vector<SpecialBlock> _special_blocks;
      std::vector<IdxType> _group_id_to_special_block;
      std::vector<uint8_t> _group_is_special_block_root;
      std::vector<uint8_t> _group_is_trivial_special_block_root;
      std::vector<IdxType> _point_to_special_block;
      std::vector<uint8_t> _point_is_special_block_root;
      std::vector<std::vector<SpecialEdge>> _special_edges_by_point;
      std::vector<std::vector<SpecialEdge>> _special_heavy_edges_by_point;
      SpecialBlockBuildSummary _special_block_summary;
      void build_special_blocks();
      void rebuild_special_block_indexes();
      void build_special_edge_overlay();
      void save_special_blocks(const std::string &prefix) const;
      void load_special_blocks(const std::string &prefix, const std::map<std::string, std::string> &meta_data);
      bool is_trivial_special_block_root_group(IdxType group_id) const;
      bool is_special_block_root_group(IdxType group_id) const;
      bool is_special_block_root_point(IdxType point_id) const;
      bool query_covers_special_block(const std::vector<LabelType> &query_labels,
                                      const SpecialBlock &block) const;
      void populate_special_query_stats(const std::vector<LabelType> &query_labels,
                                        QueryStats &stats) const;

      // Label navigating graph and coverage metadata.
      std::shared_ptr<LabelNavGraph> _label_nav_graph = nullptr;
      void get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                              bool avoid_self = false, bool need_containment = true);
      void get_min_super_sets_optimized_bucket(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                               bool avoid_self = false, bool need_containment = true);
      void get_min_super_sets_original_sort(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                            bool avoid_self = false, bool need_containment = true);
      void cal_f_coverage_ratio();
      void build_label_nav_graph();
      void build_label_nav_graph_optimized_phase1();
      void build_label_nav_graph_legacy_allocating();
      void build_label_nav_graph_original_cpu();
      size_t count_all_descendants(IdxType group_id) const;
      void print_lng_descendants_num(const std::string &filename) const;
      void get_descendants_info();
      void get_descendants_info_optimized_epoch_bfs();
      void get_descendants_info_legacy_hash_bfs();

      // Group storage and ID mappings.
      std::vector<IdxType> _new_to_old_vec_ids;
      std::vector<IdxType> _old_to_new_vec_ids;
      std::vector<std::pair<IdxType, IdxType>> _group_id_to_range;
      std::vector<std::shared_ptr<IStorage>> _group_storages;
      void prepare_group_storages_graphs();
      void reserve_graph_neighbor_capacity();

      // Group graph construction. CPU/Tagore/FastGrnnd implementations live in
      // uni_nav_graph_group_graph.cpp.
      std::string _index_name;
      std::vector<std::shared_ptr<Graph>> _group_graphs;
      std::vector<std::shared_ptr<Graph>> _special_block_target_graphs;
      std::vector<IdxType> _group_entry_points;
      bool _intra_group_graph_ids_are_global = false;
      bool should_write_intra_group_global_ids() const;
      void build_graph_for_all_groups();
      void build_graph_for_all_groups_tagore_cuda();
      void build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points, IdxType base_offset = 0);
      void build_bounded_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points, IdxType max_degree, IdxType base_offset = 0);
      TagoreGroupPartition partition_tagore_groups(const TagoreGroupBuildContext &context) const;
      TagoreFallbackBuildStats build_tagore_fallback_groups(const std::vector<IdxType> &fallback_group_ids,
                                                            const TagoreGroupBuildContext &context);
      TagoreBatchBuildArtifacts build_tagore_batch_artifacts(const std::vector<TagoreGroupRequest> &tagore_requests,
                                                             const TagoreGroupBuildContext &context);
      void fill_tagore_batch_results(const std::vector<IdxType> &tagore_group_ids,
                                     const std::vector<TagoreBuildResult> &tagore_results,
                                     const std::vector<size_t> &exact_packed_rank,
                                     const std::vector<uint32_t> &exact_packed_graph,
                                     const std::vector<uint32_t> &exact_packed_offsets,
                                     uint32_t exact_packed_stride,
                                     const TagoreGroupBuildContext &context,
                                     TagoreBuildResult &timing_acc,
                                     double &fill_wall_ms,
                                     double &fill_cpu_sum_ms);
      std::vector<std::shared_ptr<Vamana>> _vamana_instances;

      // Legacy global_graph placeholder persisted for index format
      // compatibility. Current build/search paths do not construct or query a
      // global Vamana graph.
      std::shared_ptr<Graph> _global_graph;
      IdxType _global_vamana_entry_point = 0;

      // Vector-attribute graph and persistence helpers.
      void build_vector_and_attr_graph();
      size_t count_graph_edges() const;
      void save_bipartite_graph(const std::string &filename);
      uint32_t compute_checksum() const;

      // LNG descendant and coverage caches used by filtering and bitmap helpers.
      std::vector<BitsetType> _lng_descendants_bits;
      std::vector<BitsetType> _covered_sets_bits;
      std::vector<roaring::Roaring> _lng_descendants_rb;
      std::vector<roaring::Roaring> _covered_sets_rb;
      std::unique_ptr<GpuCoverFrontierProvider> _gpu_cover_frontier_provider;
      std::mutex _gpu_cover_frontier_provider_mutex;
      void initialize_lng_descendants_coverage_bitsets();
      void initialize_roaring_bitsets();

      // Legacy distance-oriented edge augmentation.
      void add_new_distance_oriented_edges(
          const std::string &dataset,
          uint32_t num_threads,
          ANNS::AcornInUng new_cross_edge);
      int _num_distance_oriented_edges = 0;
      std::unordered_set<uint64_t> _my_new_edges_set;

      void finalize_intra_group_graphs();

      // index parameters for each graph
      IdxType _max_degree = 0;
      IdxType _Lbuild = 0;
      float _alpha = 0.0f;
      uint32_t _num_threads = 0;
      std::string _scenario;

      // Cross-group edge orchestration. CPU/GPU route adapters and graph
      // writeback live in uni_nav_graph_cross_edges.cpp.
      IdxType _num_cross_edges = 0;
      UngBuildConfig _build_config;
      std::vector<SearchQueue> _cross_group_neighbors;
      void build_cross_group_edges();
      void build_cross_group_edges_original_cpu();
      CrossEdgeBackend resolve_cross_edge_backend() const;
      void build_cross_edges_generate_cpu_baseline(std::vector<SearchQueue> &cross_group_neighbors,
                                                   SearchCacheList &search_cache_list);
      void build_cross_edges_generate_cpu_exact_scan(std::vector<SearchQueue> &cross_group_neighbors);
      void build_cross_edges_generate_cpu_hybrid_scan_vamana(std::vector<SearchQueue> &cross_group_neighbors,
                                                             SearchCacheList &search_cache_list);
      void append_cross_edges_from_target_range(IdxType source_point_id,
                                                IdxType target_first,
                                                IdxType target_second,
                                                std::shared_ptr<Vamana> target_index,
                                                SearchCacheList *search_cache_list,
                                                bool use_exact_scan,
                                                SearchQueue &out_neighbors) const;
      void append_cross_edges_from_target_points(IdxType source_point_id,
                                                 const std::vector<IdxType> &target_point_ids,
                                                 std::shared_ptr<Vamana> target_index,
                                                 SearchCacheList *search_cache_list,
                                                 bool use_exact_scan,
                                                 SearchQueue &out_neighbors) const;
      bool build_cross_edges_generate_gpu_optimized(std::vector<SearchQueue> &cross_group_neighbors,
                                                    const CrossEdgeGpuRuntimeConfig &gpu_route,
                                                    std::vector<std::vector<IdxType>> *cross_group_neighbor_ids,
                                                    std::vector<IdxType> *cross_group_neighbor_flat_ids,
                                                    CrossEdgeBuildTiming &timing);
      bool build_cross_edges_generate_cuvs_bruteforce(std::vector<SearchQueue> &cross_group_neighbors,
                                                      CrossEdgeBuildTiming &timing,
                                                      const CrossEdgeGpuRuntimeConfig &gpu_route);
      bool build_cross_edges_generate_gpu_source_exact(std::vector<SearchQueue> &cross_group_neighbors,
                                                       std::vector<std::vector<IdxType>> *cross_group_neighbor_ids,
                                                       std::vector<IdxType> *cross_group_neighbor_flat_ids,
                                                       CrossEdgeBuildTiming &timing,
                                                       const CrossEdgeGpuRuntimeConfig &gpu_route);
      void build_cross_edges_generate_additional(
          std::vector<std::vector<std::pair<IdxType, IdxType>>> *materialized,
          const CrossEdgeHostOutputView &cross_edge_outputs,
          SearchCacheList *search_cache_list,
          CrossEdgeBuildTiming &timing);
      void merge_cross_edges_to_graph(const CrossEdgeHostOutputView &cross_edge_outputs,
                                      CrossEdgeBuildTiming &timing);
      void merge_additional_edges_to_graph(
          const std::vector<std::vector<std::pair<IdxType, IdxType>>> &additional_edges,
          CrossEdgeBuildTiming &timing);

      // CUDA cross-edge backend entry points. These preserve CPU cross-edge
      // semantics while each GPU route chooses its own packing, execution, and
      // writeback strategy internally.
      void gpu_prepare_all_vectors_on_device(const CrossEdgeVectorUploadConfig &upload_config,
                                             double *h2d_ms = nullptr,
                                             double *d2h_ms = nullptr);
      void gpu_prepare_all_vectors_for_cross_edge(CrossEdgeBuildTiming &timing,
                                                  const CrossEdgeGpuRuntimeConfig &gpu_route);
      void gpu_release_all_vectors_on_device();
      void gpu_cross_groups_search_all_batched(
          const std::vector<IdxType> &target_group_ids,
          int dim,
          int topk,
          std::vector<SearchQueue> &cross_group_neighbors,
          std::vector<std::vector<IdxType>> *cross_group_neighbor_ids,
          std::vector<IdxType> *cross_group_neighbor_flat_ids,
          CrossEdgeBuildTiming *timing,
          const CrossEdgeGpuRuntimeConfig &gpu_route);
      void gpu_cross_groups_search_x_streaming(
          const std::vector<IdxType> &target_group_ids,
          int dim,
          int topk,
          std::vector<SearchQueue> &cross_group_neighbors,
          CrossEdgeBuildTiming *timing,
          const CrossEdgeGpuRuntimeConfig &gpu_route);
      bool gpu_build_special_inter_edges(
          const std::vector<SpecialBlock> &special_blocks,
          const std::vector<std::vector<IdxType>> &block_points,
          std::vector<std::vector<SpecialEdge>> &special_edges_by_point,
          IdxType &inter_edge_count,
          double &gpu_ms);

      // obtain the final unified navigating graph
      void add_offset_for_uni_nav_graph();

      // statistics
      float _index_time = 0, _label_processing_time = 0, _build_graph_time = 0, _build_vector_attr_graph_time = 0, _cal_descendants_time = 0, _cal_coverage_ratio_time = 0;
      float _build_LNG_time = 0, _build_cross_edges_time = 0;
      double _build_roaring_bitsets_time = 0.0;
      float _index_size = 0.0f, _index_size_add_rb = 0.0f;
      IdxType _graph_num_edges = 0, _LNG_num_edges = 0;

      double _tagore_groups = 0.0;
      double _tagore_points = 0.0;
      double _tagore_direct_build_wall_time_ms = 0.0;
      double _tagore_no_alloc_build_time_ms = 0.0;
      double _tagore_pack_time_ms = 0.0;
      double _tagore_convert_time_ms = 0.0;
      double _tagore_workspace_alloc_time_ms = 0.0;
      double _tagore_h2d_time_ms = 0.0;
      double _tagore_memset_time_ms = 0.0;
      double _tagore_gnn_time_ms = 0.0;
      double _tagore_prune_time_ms = 0.0;
      double _tagore_grnnd_refine_time_ms = 0.0;
      double _tagore_d2h_time_ms = 0.0;
      double _tagore_fill_time_ms = 0.0;
      double _tagore_workspace_free_time_ms = 0.0;
      double _tagore_unaccounted_time_ms = 0.0;
      double _tagore_h2d_effective_gbps = 0.0;
      double _tagore_d2h_effective_gbps = 0.0;
      double _tagore_gnn_mpts_s = 0.0;
      double _tagore_prune_mpts_s = 0.0;

      // Detailed timing for legacy ACORN-based distance-oriented edge augmentation.
      double _cross_edge_step1_time_ms = 0.0;
      double _cross_edge_step2_acorn_time_ms = 0.0;
      double _cross_edge_step3_add_dist_edges_time_ms = 0.0;
      double _cross_edge_step4_add_hierarchy_edges_time_ms = 0.0;
      void statistics();

      std::string _dataset;

      // idea1 selector
      std::unique_ptr<MethodSelector> _trie_method_selector;
      TrieStaticMetrics _trie_static_metrics;

      // idea2 selector
      std::shared_ptr<faiss::IndexACORNFlat> _acorn_index;
      std::shared_ptr<faiss::IndexACORNFlat> _acorn_1_index;
      std::unique_ptr<MethodSelector> _ung_acorn_selector;
   };
}

#endif // UNG_H
