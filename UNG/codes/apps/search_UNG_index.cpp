#include <chrono>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <bitset>
#include <map>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <vector>

#include <omp.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "uni_nav_graph.h"
#include "navix_index.h"
#include "utils.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace
{

struct BitmapComparisonTiming
{
   double ung_ms = 0.0;
   double attr_ms = 0.0;
   bool skipped = false;
};

float calculate_single_query_recall(const std::pair<ANNS::IdxType, float> *gt,
                                    const std::pair<ANNS::IdxType, float> *results,
                                    ANNS::IdxType K)
{
   std::unordered_set<ANNS::IdxType> gt_set;
   for (int i = 0; i < K; ++i)
   {
      if (gt[i].first != -1)
      {
         gt_set.insert(gt[i].first);
      }
   }

   int correct = 0;
   for (int i = 0; i < K; ++i)
   {
      if (results[i].first != -1 && gt_set.count(results[i].first))
      {
         correct++;
      }
   }

   return static_cast<float>(correct) / gt_set.size();
}

double query_block_authorization_time_ms(const ANNS::QueryStats &stats)
{
   // Includes the exact upper-layer route gate and, when the overlay is used,
   // the subsequent per-block coverage propagation.
   return stats.special_cover_time_ms;
}

double query_graph_search_time_ms(const ANNS::QueryStats &stats)
{
   // The special-block backend starts core_search_time_ms before it converts
   // entry groups into scored entry points.  Remove that initialization phase
   // to expose the time spent in the subsequent graph search.
   const double graph_search = stats.core_search_time_ms - stats.entry_point_setup_time_ms;
   return graph_search > 0.0 ? graph_search : 0.0;
}

double query_residual_time_ms(const ANNS::QueryStats &stats)
{
   // core_search_time_ms contains entry-point setup but starts after Special
   // Block authorization.  The four reported phases therefore close exactly
   // against the per-query total without double-counting entry work.
   const double residual = stats.time_ms - stats.get_min_super_sets_time_ms -
                           query_block_authorization_time_ms(stats) -
                           stats.core_search_time_ms;
   return residual > 0.0 ? residual : 0.0;
}

size_t query_graph_search_distance_calcs(const ANNS::QueryStats &stats)
{
   // Every newly visited graph node is scored once.  Clamp defensively for
   // alternate backends whose counters may use a different convention.
   return std::min(stats.num_nodes_visited, stats.num_distance_calcs);
}

size_t query_entry_point_distance_calcs(const ANNS::QueryStats &stats)
{
   // num_distance_calcs consists of the initial entry-point scores followed by
   // one score for every newly visited graph node.
   return stats.num_distance_calcs - query_graph_search_distance_calcs(stats);
}

std::vector<std::vector<ANNS::IdxType>> precompute_entry_group_ids(
    const ANNS::UniNavGraph &index,
    const std::shared_ptr<ANNS::IStorage> &query_storage)
{
   const auto num_queries = query_storage->get_num_points();
   std::vector<std::vector<ANNS::IdxType>> all_entry_group_ids(num_queries);

   std::cout << "\n--- Step 1: Pre-computing Entry Group IDs (Measuring Entry Cost) ---" << std::endl;
   auto entry_cost_start_time = std::chrono::high_resolution_clock::now();
   static std::atomic<int> trie_debug_print_counter{0};
#pragma omp parallel for
   for (long long id = 0; id < static_cast<long long>(num_queries); ++id)
   {
      const auto query_id = static_cast<ANNS::IdxType>(id);
      const auto &query_labels = query_storage->get_label_set(query_id);
      ANNS::QueryStats dummy_stats;
      index.get_min_super_sets_debug(
          query_labels,
          all_entry_group_ids[query_id],
          false,
          true,
          trie_debug_print_counter,
          false,
          false,
          dummy_stats,
          false);
   }
   const double entry_cost_total_time =
       std::chrono::duration<double, std::milli>(
           std::chrono::high_resolution_clock::now() - entry_cost_start_time)
           .count();
   std::cout << "Total time for finding all entry groups (Entry Cost): "
             << entry_cost_total_time << " ms\n"
             << std::endl;
   return all_entry_group_ids;
}

BitmapComparisonTiming run_bitmap_comparison(
    const ANNS::UniNavGraph &index,
    const std::shared_ptr<ANNS::IStorage> &query_storage,
    const std::vector<std::vector<ANNS::IdxType>> &all_entry_group_ids)
{
   BitmapComparisonTiming timing;
   const bool skip_bitmap_compare = std::getenv("UNG_SEARCH_SKIP_BITMAP_COMPARE") != nullptr;
   if (skip_bitmap_compare)
   {
      timing.skipped = true;
      std::cout << "--- Step 2: Skipping Fair Bitmap Computation Comparison "
                   "(UNG_SEARCH_SKIP_BITMAP_COMPARE=1) ---\n"
                << std::endl;
      return timing;
   }

   const auto num_queries = query_storage->get_num_points();
   std::cout << "--- Step 2: Starting Fair Bitmap Computation Comparison ---" << std::endl;
   {
      std::cout << "  -> Testing UNG method (compute_bitmap_from_groups)..." << std::endl;
      std::vector<roaring::Roaring> ung_bitmaps(num_queries);
      auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
      for (long long id = 0; id < static_cast<long long>(num_queries); ++id)
      {
         const auto query_id = static_cast<ANNS::IdxType>(id);
         ung_bitmaps[query_id] = index.compute_bitmap_from_groups(all_entry_group_ids[query_id]);
      }
      timing.ung_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - start_time)
              .count();
      std::cout << "  -> UNG bitmap generation time: " << timing.ung_ms << " ms" << std::endl;
   }
   {
      std::cout << "  -> Testing Attribute method (compute_attribute_bitmap)..." << std::endl;
      std::vector<std::bitset<10000001>> attr_bitmaps(num_queries);
      auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
      for (long long id = 0; id < static_cast<long long>(num_queries); ++id)
      {
         const auto query_id = static_cast<ANNS::IdxType>(id);
         attr_bitmaps[query_id] = index.compute_attribute_bitmap(query_storage->get_label_set(query_id)).first;
      }
      timing.attr_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - start_time)
              .count();
      std::cout << "  -> Attribute bitmap generation time: " << timing.attr_ms << " ms" << std::endl;
   }
   std::cout << "--- Fair Comparison Finished ---\n"
             << std::endl;
   return timing;
}

} // namespace

int main(int argc, char **argv)
{
   std::string data_type, dist_fn, scenario;
   std::string base_bin_file, query_bin_file, base_label_file, query_label_file, gt_file, index_path_prefix, result_path_prefix, selector_model_prefix, selector_model_prefix_legacy, query_group_id_file;
   std::string acorn_index_path, acorn_1_index_path, block_index_path_prefix;
   std::string entry_group_strategy_arg = "optimized_lng";
   std::string legacy_entry_group_provider_arg;
   std::string graph_search_backend_arg = "neighbor_list";
   ANNS::EntryGroupStrategy entry_group_strategy = ANNS::EntryGroupStrategy::OptimizedLng;
   ANNS::SearchGraphBackendImpl graph_search_backend = ANNS::SearchGraphBackendImpl::NeighborList;
   ANNS::IdxType K, num_entry_points;
   std::vector<ANNS::IdxType> Lsearch_list;
   uint32_t num_threads;
   bool is_new_method = false;                                 // true: use new method
   bool is_idea2_available = false;                            // true: use original ung
   bool is_new_trie_method = false, is_rec_more_start = false; // false:默认的UNG原始trie tree方法,true：递归；false:默认的root
   bool is_ung_more_entry = false;                             // false:默认的UNG原始entry point选择方法,true：更多entry points
   bool skip_query_features = false;
   bool skip_bitmap_comparison = false;
   int num_repeats = 1;                                        // 默认重复1次
   int force_use_alg = 0;                                      // 0: auto, 1: UNG (nT=false), 2: UNG-nTtrue, 3/4: ACORN, 5: NaviX
   int lsearch_start, lsearch_step;
   int efs_start, efs_step_slow, efs_step_fast, lsearch_threshold;
   std::string dataset; 

   try
   {
      po::options_description desc{"Arguments"};
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                         "dataset");
      desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                         "data type <int8/uint8/float>");
      desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                         "distance function <L2/IP/cosine>");
      desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                         "File containing the base vectors in binary format");
      desc.add_options()("query_bin_file", po::value<std::string>(&query_bin_file)->required(),
                         "File containing the query vectors in binary format");
      desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->default_value(""),
                         "Base label file in txt format");
      desc.add_options()("query_label_file", po::value<std::string>(&query_label_file)->default_value(""),
                         "Query label file in txt format");
      desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                         "Filename for the computed ground truth in binary format");
      desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(),
                         "Number of ground truth nearest neighbors to compute");
      desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS),
                         "Number of threads to use");
      desc.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
                         "Path to save the querying result file");
      desc.add_options()("selector_model_prefix", po::value<std::string>(&selector_model_prefix),
                         "Path to selector model directory");
      desc.add_options()("selector_modle_prefix", po::value<std::string>(&selector_model_prefix_legacy),
                         "Deprecated alias for selector_model_prefix");
      desc.add_options()("query_group_id_file", po::value<std::string>(&query_group_id_file)->default_value(""),
                         "query_group_id_file");
      desc.add_options()("acorn_index_path", po::value<std::string>(&acorn_index_path)->default_value(""),
                         "acorn_index_path");
      desc.add_options()("acorn_1_index_path", po::value<std::string>(&acorn_1_index_path)->default_value(""),
                         "acorn_1_index_path");
      desc.add_options()("block_index_path_prefix", po::value<std::string>(&block_index_path_prefix)->default_value(""),
                         "Optional path prefix of an independent Trie block index");

      // graph search parameters
      desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("containment"),
                         "Scenario for building UniNavGraph, <equality/containment/overlap/nofilter>");
      desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                         "Prefix of the path to load the index");
      desc.add_options()("num_entry_points", po::value<ANNS::IdxType>(&num_entry_points)->default_value(ANNS::default_paras::NUM_ENTRY_POINTS),
                         "Number of entry points in each entry group");
      desc.add_options()("Lsearch", po::value<std::vector<ANNS::IdxType>>(&Lsearch_list)->multitoken()->required(),
                         "Number of candidates to search in the graph");
      desc.add_options()("is_new_method", po::value<bool>(&is_new_method)->required(),
                         "is_new_method");
      desc.add_options()("is_idea2_available", po::value<bool>(&is_idea2_available)->required(),
                         "is_idea2_available");
      desc.add_options()("is_new_trie_method", po::value<bool>(&is_new_trie_method)->required(),
                         "is_new_trie_method");
      desc.add_options()("is_rec_more_start", po::value<bool>(&is_rec_more_start)->required(),
                         "is_rec_more_start");
      desc.add_options()("is_ung_more_entry", po::value<bool>(&is_ung_more_entry)->default_value(false),
                         "Enable expanded/oracle-assisted UNG entry group selection. "
                         "When true, query_group_id_file can affect entry groups.");
      desc.add_options()("skip_query_features", po::value<bool>(&skip_query_features)->default_value(false),
                         "Skip query feature CSV generation before search.");
      desc.add_options()("skip_bitmap_comparison", po::value<bool>(&skip_bitmap_comparison)->default_value(false),
                         "Skip pre-search bitmap comparison diagnostics.");
      desc.add_options()("num_repeats", po::value<int>(&num_repeats)->default_value(1),
                         "Number of repeats for each Lsearch value");
      desc.add_options()("force_use_alg", po::value<int>(&force_use_alg)->required(),
                         "force_use_alg: 0 auto, 1 UNG, 2 UNG new trie, 3/4 ACORN, 5 NaviX adaptive-local");
      desc.add_options()("lsearch_start", po::value<int>(&lsearch_start)->required(), "Lsearch start value");
      desc.add_options()("lsearch_step", po::value<int>(&lsearch_step)->required(), "Lsearch step value");
      desc.add_options()("efs_start", po::value<int>(&efs_start)->required(), "ACORN efs start value");
      desc.add_options()("efs_step_slow", po::value<int>(&efs_step_slow)->required(), "ACORN efs step value");
      desc.add_options()("efs_step_fast", po::value<int>(&efs_step_fast)->required(), "ACORN efs step value");
      desc.add_options()("lsearch_threshold", po::value<int>(&lsearch_threshold)->required(), "lsearch_threshold");
      desc.add_options()("entry_group_strategy", po::value<std::string>(&entry_group_strategy_arg)->default_value("optimized_lng"),
                         "Entry-group strategy: original, optimized_lng, or trie.");
      desc.add_options()("entry_group_provider", po::value<std::string>(&legacy_entry_group_provider_arg),
                         "Deprecated alias for entry_group_strategy.");
      desc.add_options()("graph_search_backend", po::value<std::string>(&graph_search_backend_arg)->default_value("neighbor_list"),
                         "UNG graph search backend: neighbor_list/graph/default/0 or csr/1.");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help"))
      {
         std::cout << desc;
         return 0;
      }
      po::notify(vm);
      if (!legacy_entry_group_provider_arg.empty())
      {
         if (!vm["entry_group_strategy"].defaulted() &&
             ANNS::parse_entry_group_strategy(entry_group_strategy_arg) !=
                 ANNS::parse_entry_group_strategy(legacy_entry_group_provider_arg))
            throw std::invalid_argument(
                "entry_group_strategy and deprecated entry_group_provider disagree");
         entry_group_strategy_arg = legacy_entry_group_provider_arg;
      }
      if (selector_model_prefix.empty())
         selector_model_prefix = selector_model_prefix_legacy;
      if (force_use_alg != 5 && selector_model_prefix.empty())
      {
         std::cerr << "Missing required option: selector_model_prefix (or deprecated selector_modle_prefix)" << std::endl;
         return -1;
      }
      if (force_use_alg != 5 && query_group_id_file.empty())
      {
         std::cerr << "Missing required option: query_group_id_file" << std::endl;
         return -1;
      }
      entry_group_strategy = ANNS::parse_entry_group_strategy(entry_group_strategy_arg);
      graph_search_backend = ANNS::parse_search_graph_backend_impl(graph_search_backend_arg);
   }
   catch (const std::exception &ex)
   {
      std::cerr << ex.what() << std::endl;
      return -1;
   }

   // check scenario
   if (scenario != "containment" && scenario != "equality" && scenario != "overlap")
   {
      std::cerr << "Invalid scenario: " << scenario << std::endl;
      return -1;
   }

   // load query data
   std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
   query_storage->load_from_file(query_bin_file, query_label_file);

   // 加载查询来源组ID文件。Standalone NaviX 不使用 UNG entry groups。
   std::vector<ANNS::IdxType> true_query_group_ids;
   if (force_use_alg != 5)
   {
      std::ifstream source_group_file(query_group_id_file);
      if (source_group_file.is_open())
      {
         ANNS::IdxType group_id;
         while (source_group_file >> group_id)
         {
            true_query_group_ids.push_back(group_id);
         }
         source_group_file.close();
         std::cout << "成功加载 " << true_query_group_ids.size() << " 个查询的来源组ID。" << std::endl;
      }
      else // 即使没找到，程序也可以继续，只是没有优化效果
      {
         std::cerr << "警告：未找到查询来源组ID文件: " << query_group_id_file << std::endl;
      }
   }

   // preparation
   auto num_queries = query_storage->get_num_points();
   std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
   auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
   ANNS::load_gt_file(gt_file, gt, num_queries, K);
   auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];

   if (force_use_alg == 5)
   {
      fs::create_directories(result_path_prefix);
      ANNS::StandaloneNavixIndex navix_index;
      navix_index.load(index_path_prefix, data_type);

      struct SearchTimeLog
      {
         int repeat;
         ANNS::IdxType l_search;
         int efs;
         double time_ms;
         float avg_recall;
      };
      std::vector<SearchTimeLog> detailed_times;
      std::map<ANNS::IdxType, std::vector<double>> time_per_lsearch;
      std::map<ANNS::IdxType, std::vector<float>> recall_per_lsearch;
      std::map<ANNS::IdxType, std::vector<int>> efs_per_lsearch;
      std::vector<std::vector<std::vector<ANNS::QueryStats>>> query_stats(
          num_repeats,
          std::vector<std::vector<ANNS::QueryStats>>(
              Lsearch_list.size(), std::vector<ANNS::QueryStats>(num_queries)));

      for (int repeat = 0; repeat < num_repeats; ++repeat)
      {
         std::cout << "\n=== NaviX Repeat " << (repeat + 1) << "/" << num_repeats << " ===" << std::endl;
         for (size_t LsearchId = 0; LsearchId < Lsearch_list.size(); ++LsearchId)
         {
            const ANNS::IdxType current_Lsearch = Lsearch_list[LsearchId];
            std::vector<float> num_cmps(num_queries);
            auto start_time = std::chrono::high_resolution_clock::now();
            navix_index.search(query_storage, distance_handler, num_threads, current_Lsearch, K,
                               results, num_cmps, query_stats[repeat][LsearchId]);
            const double time_cost = std::chrono::duration<double, std::milli>(
                                         std::chrono::high_resolution_clock::now() - start_time)
                                         .count();

            double total_recall_for_batch = 0.0;
            for (ANNS::IdxType i = 0; i < num_queries; ++i)
            {
               query_stats[repeat][LsearchId][i].recall =
                   calculate_single_query_recall(gt + i * K, results + i * K, K);
               total_recall_for_batch += query_stats[repeat][LsearchId][i].recall;
            }
            const float avg_recall_for_batch =
                (num_queries > 0) ? (static_cast<float>(total_recall_for_batch) / num_queries) : 0.0f;

            detailed_times.push_back({repeat, current_Lsearch, 0, time_cost, avg_recall_for_batch});
            time_per_lsearch[current_Lsearch].push_back(time_cost);
            recall_per_lsearch[current_Lsearch].push_back(avg_recall_for_batch);
            efs_per_lsearch[current_Lsearch].push_back(0);
            std::cout << "  NaviX Lsearch=" << current_Lsearch << ", time=" << time_cost
                      << "ms, avg_recall=" << avg_recall_for_batch << std::endl;
         }
      }

      std::ofstream details_out(result_path_prefix + "search_time_details.csv");
      details_out << "Repeat,Lsearch,efs,Time_ms,Avg_Recall\n";
      for (const auto &log : detailed_times)
         details_out << log.repeat << "," << log.l_search << "," << log.efs << ","
                     << log.time_ms << "," << log.avg_recall << "\n";

      std::ofstream summary_out(result_path_prefix + "search_time_summary.csv");
      summary_out << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,"
                  << "Average_EntryGroupSearchTime_ms,Average_EntryPointSetupTime_ms,"
                  << "Average_GraphSearchTime_ms,Average_OtherTime_ms,"
                  << "Average_RegularEdgesScanned,Average_FreeEdgesScanned,"
                  << "Average_SpecialIntraEdgesScanned,Average_SpecialInterEdgesScanned,"
                  << "Average_TotalEdgesScanned,Average_NodesVisited,Average_TotalDistanceCalcs,"
                  << "Average_EntryPointDistanceCalcs,Average_GraphSearchDistanceCalcs\n";
      for (size_t LsearchId = 0; LsearchId < Lsearch_list.size(); ++LsearchId)
      {
         const ANNS::IdxType l_search = Lsearch_list[LsearchId];
         const auto &times = time_per_lsearch.at(l_search);
         const double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
         const auto &recalls = recall_per_lsearch.at(l_search);
         const double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
         double entry_group_search_time = 0.0;
         double entry_point_setup_time = 0.0;
         double graph_search_time = 0.0;
         double other_time = 0.0;
         double regular_edges = 0.0;
         double free_edges = 0.0;
         double total_distance_calcs = 0.0;
         double entry_point_distance_calcs = 0.0;
         double graph_search_distance_calcs = 0.0;
         size_t sample_count = 0;
         for (int repeat = 0; repeat < num_repeats; ++repeat)
         {
            for (ANNS::IdxType query_id = 0; query_id < num_queries; ++query_id)
            {
               const auto &stats = query_stats[repeat][LsearchId][query_id];
               entry_group_search_time += stats.get_min_super_sets_time_ms;
               entry_point_setup_time += stats.entry_point_setup_time_ms;
               graph_search_time += query_graph_search_time_ms(stats);
               other_time += query_residual_time_ms(stats);
               regular_edges += stats.regular_edges_scanned;
               free_edges += stats.free_edges_scanned;
               total_distance_calcs += stats.num_distance_calcs;
               entry_point_distance_calcs += query_entry_point_distance_calcs(stats);
               graph_search_distance_calcs += query_graph_search_distance_calcs(stats);
               ++sample_count;
            }
         }
         const double divisor = sample_count > 0 ? static_cast<double>(sample_count) : 1.0;
         summary_out << l_search << ",0," << avg_time << "," << avg_recall << ","
                     << entry_group_search_time / divisor << ","
                     << entry_point_setup_time / divisor << ","
                     << graph_search_time / divisor << ","
                     << other_time / divisor << ","
                     << regular_edges / divisor << ","
                     << free_edges / divisor << ","
                     << (regular_edges + free_edges) / divisor << ","
                     << total_distance_calcs / divisor << ","
                     << entry_point_distance_calcs / divisor << ","
                     << graph_search_distance_calcs / divisor << "\n";
      }

      std::ofstream detail_out(result_path_prefix + "query_details_repeat" + std::to_string(num_repeats) + ".csv");
      detail_out << "Lsearch,QueryID,Time_ms,core_search_time_ms,ELS_time_ms,OtherT_ms,Recall,"
                 << "DistCalcs,NumNodeVisited,QuerySize,CandSize,NumEntries\n";
      for (int repeat = 0; repeat < num_repeats; ++repeat)
      {
         for (size_t LsearchId = 0; LsearchId < Lsearch_list.size(); ++LsearchId)
         {
            for (ANNS::IdxType i = 0; i < num_queries; ++i)
            {
               const auto &stats = query_stats[repeat][LsearchId][i];
               detail_out << Lsearch_list[LsearchId] << "," << i << ","
                          << stats.time_ms << "," << stats.core_search_time_ms << ","
                          << stats.get_min_super_sets_time_ms << "," << query_residual_time_ms(stats) << ","
                          << stats.recall << "," << stats.num_distance_calcs << ","
                          << stats.num_nodes_visited << "," << stats.query_length << ","
                          << stats.candidate_set_size << "," << stats.num_entry_points << "\n";
            }
         }
      }

      delete[] results;
      delete[] gt;
      std::cout << "- NaviX standalone search done" << std::endl;
      return 0;
   }

   // load UNG index
   ANNS::UniNavGraph index(query_storage->get_num_points());
   index.load(index_path_prefix, selector_model_prefix, data_type, acorn_index_path, acorn_1_index_path,dataset);
   if (!block_index_path_prefix.empty())
      index.load_special_block_index(block_index_path_prefix);
   index.load_bipartite_graph(index_path_prefix + "vector_attr_graph");

   if (!skip_bitmap_comparison)
   {
      std::vector<std::vector<ANNS::IdxType>> all_entry_group_ids =
          precompute_entry_group_ids(index, query_storage);
      const BitmapComparisonTiming bitmap_timing =
          run_bitmap_comparison(index, query_storage, all_entry_group_ids);
      (void)bitmap_timing;
   }
   else
   {
      std::cout << "Skipping pre-search bitmap comparison diagnostics." << std::endl;
   }

   // calculate query features and save to CSV
   if (!skip_query_features)
   {
      std::string features_csv_path = result_path_prefix + "query_features.csv";
      index.calculate_query_features_only(
          query_storage,
          num_threads,
          features_csv_path,
          true, // is_new_trie_method
          true  // is_rec_more_start
      );
   }
   else
   {
      std::cout << "Skipping query feature CSV generation." << std::endl;
   }

   // Warm up only the route selectors. Entry-group strategies are measured as
   // part of the end-to-end query path.
   std::cout << "\n--- Starting Warm-up Phase ---" << std::endl;
   index.warmup_selectors(num_threads);
   std::cout << "--- Warm-up Finished ---"<< std::endl;

   if (Lsearch_list.empty())
   {
      std::cerr << "Lsearch list must not be empty" << std::endl;
      return -1;
   }
   const ANNS::IdxType max_lsearch = *std::max_element(Lsearch_list.begin(), Lsearch_list.end());
   std::unique_ptr<ANNS::SearchExecutionContext> search_execution_context;
   if (!ANNS::ung_env_flag_enabled("UNG_DISABLE_SEARCH_CONTEXT_REUSE"))
      search_execution_context = index.create_search_execution_context(num_threads, max_lsearch);

   // init query stats
   std::vector<std::vector<std::vector<ANNS::QueryStats>>> query_stats(num_repeats, std::vector<std::vector<ANNS::QueryStats>>(Lsearch_list.size(), std::vector<ANNS::QueryStats>(num_queries))); //(repeat,Lsearch,queryID)

   // 结构体，用于存储每一次的详细耗时
   struct SearchTimeLog
   {
      int repeat;
      ANNS::IdxType l_search;
      int efs;
      double time_ms;
      float avg_recall;
   };
   std::vector<SearchTimeLog> detailed_times;                      // 存储所有详细耗时记录
   std::map<ANNS::IdxType, std::vector<double>> time_per_lsearch;  // 使用 map 来按 Lsearch 值分组存储每次 repeat 的耗时，方便后续计算平均值
   std::map<ANNS::IdxType, std::vector<float>> recall_per_lsearch; // 用于存储每个 Lsearch 的 recall 值
   std::map<ANNS::IdxType, std::vector<int>> efs_per_lsearch;      // 用于存储每个 Lsearch 的 efs 值
   struct FilterValidationLog
   {
      int repeat;
      ANNS::IdxType l_search;
      size_t checked_results;
      size_t violations;
   };
   const bool validate_filter_results =
       ANNS::ung_env_flag_enabled("UNG_VALIDATE_FILTER_RESULTS");
   std::vector<FilterValidationLog> filter_validation_logs;

   for (int repeat = 0; repeat < num_repeats; ++repeat)
   {
      std::cout << "\n=== Repeat " << (repeat + 1) << "/" << num_repeats << " ===" << std::endl;

      // search
      std::vector<float> all_cmps, all_qpss, all_recalls;
      std::vector<float> all_time_ms, all_flag_time, all_bitmap_time, all_entry_points, all_lng_descendants, all_entry_group_coverage;
      std::vector<float> all_is_global_search; // 如果需要统计全局搜索比例

      std::cout << "Start querying ..." << std::endl;
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         ANNS::IdxType current_Lsearch = Lsearch_list[LsearchId];
         std::vector<float> num_cmps(num_queries);

         // 1. 计时并执行搜索
         auto start_time = std::chrono::high_resolution_clock::now();
         if (!is_new_method)
         {
            // index.search(...);
         }
         else
         {
            ANNS::SearchRuntimeConfig runtime = ANNS::make_search_runtime_config(
                num_threads, current_Lsearch, num_entry_points, scenario, K,
                is_idea2_available, is_new_trie_method, is_rec_more_start,
                is_ung_more_entry, false, lsearch_start, lsearch_step,
                efs_start, efs_step_slow, efs_step_fast, lsearch_threshold,
                force_use_alg, entry_group_strategy, graph_search_backend);

            if (search_execution_context)
            {
               index.search_hybrid(query_storage, distance_handler, runtime, *search_execution_context,
                                   results, num_cmps, query_stats[repeat][LsearchId],
                                   true_query_group_ids);
            }
            else
            {
               index.search_hybrid(query_storage, distance_handler, runtime, results, num_cmps,
                                   query_stats[repeat][LsearchId], true_query_group_ids);
            }
         }
         auto time_cost = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();

         // 2. 计算每个独立查询的Recall
         for (int i = 0; i < num_queries; ++i)
            query_stats[repeat][LsearchId][i].recall = calculate_single_query_recall(gt + i * K, results + i * K, K);
         if (validate_filter_results)
         {
            size_t checked_results = 0;
            const size_t violations = index.count_containment_result_violations(
                query_storage, results, K, &checked_results);
            filter_validation_logs.push_back(
                {repeat, current_Lsearch, checked_results, violations});
            std::cout << "  Filter validation Lsearch=" << current_Lsearch
                      << ", checked=" << checked_results
                      << ", violations=" << violations << std::endl;
         }

         // 3. 计算当前这一个批次 (LsearchId) 的平均Recall
         double total_recall_for_batch = 0.0;
         int total_efs_for_batch = 0;
         for (int i = 0; i < num_queries; ++i){
            total_recall_for_batch += query_stats[repeat][LsearchId][i].recall;
            total_efs_for_batch += query_stats[repeat][LsearchId][i].acorn_efs_used;
         }
         float avg_recall_for_batch = (num_queries > 0) ? (static_cast<float>(total_recall_for_batch) / num_queries) : 0.0f;
         int efs_for_batch = (num_queries > 0) ? (total_efs_for_batch / num_queries) : 0;

         // 4. 将批处理时间 和 该批次的平均Recall 存入相应的数据结构中
         // a. 存入 detailed_times 用于生成 search_time_details.csv
         detailed_times.push_back({repeat, current_Lsearch, efs_for_batch,time_cost, avg_recall_for_batch});

         // b. 按 Lsearch 值分组存入 map，用于后续计算总平均值，生成 search_time_summary.csv
         time_per_lsearch[current_Lsearch].push_back(time_cost);
         recall_per_lsearch[current_Lsearch].push_back(avg_recall_for_batch);
         efs_per_lsearch[current_Lsearch].push_back(efs_for_batch);

         std::cout << "  Lsearch=" << current_Lsearch << ", efs=" << efs_per_lsearch[current_Lsearch][0] << ", time=" << time_cost << "ms" << ", avg_recall=" << avg_recall_for_batch << std::endl;

      }
   }

   if (validate_filter_results)
   {
      std::ofstream validation_out(result_path_prefix + "filter_validation.csv");
      validation_out << "Repeat,Lsearch,CheckedResults,FilterViolations\n";
      size_t total_violations = 0;
      for (const auto &row : filter_validation_logs)
      {
         validation_out << row.repeat << ',' << row.l_search << ','
                        << row.checked_results << ',' << row.violations << '\n';
         total_violations += row.violations;
      }
      if (total_violations != 0)
      {
         std::cerr << "Filtered-result validation found " << total_violations
                   << " invalid returned points." << std::endl;
         return 2;
      }
   }

   // save search_time_details.csv
   std::string details_file_path = result_path_prefix + "search_time_details.csv";
   std::ofstream details_out(details_file_path);
   if (details_out.is_open())
   {
      details_out << "Repeat,Lsearch,efs,Time_ms,Avg_Recall\n"; // <-- 修改表头
      for (const auto &log : detailed_times)
      {
         details_out << log.repeat << "," << log.l_search <<","<< log.efs << "," << log.time_ms << "," << log.avg_recall << "\n";
      }
      details_out.close();
      std::cout << "\n详细的搜索耗时已保存到: " << details_file_path << std::endl;
   }
   else
   {
      std::cerr << "错误：无法打开文件 " << details_file_path << " 进行写入" << std::endl;
   }

   // save search_time_summary.csv
   std::string summary_file_path = result_path_prefix + "search_time_summary.csv";
   std::ofstream summary_out(summary_file_path);
   if (summary_out.is_open())
   {
      summary_out << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,"
                  << "Average_EntryGroupSearchTime_ms,Average_EntryPointSetupTime_ms,"
                  << "Average_BlockAuthorizationTime_ms,Average_GraphSearchTime_ms,"
                  << "Average_ResidualTime_ms,"
                  << "Average_RegularEdgesScanned,Average_FreeEdgesScanned,"
                  << "Average_SpecialIntraEdgesScanned,Average_SpecialInterEdgesScanned,"
                  << "Average_TotalEdgesScanned,Average_NodesVisited,Average_TotalDistanceCalcs,"
                  << "Average_EntryPointDistanceCalcs,Average_GraphSearchDistanceCalcs\n";
      for (size_t LsearchId = 0; LsearchId < Lsearch_list.size(); ++LsearchId)
      {
         const ANNS::IdxType l_search = Lsearch_list[LsearchId];
         const auto &times = time_per_lsearch.at(l_search);
         if (!times.empty())
         {
            // 计算平均时间
            double sum_time = std::accumulate(times.begin(), times.end(), 0.0);
            double avg_time = sum_time / times.size();

            // 计算平均召回率
            const auto &recalls = recall_per_lsearch.at(l_search);
            double sum_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0f);
            double avg_recall = sum_recall / recalls.size();

            // 2. 新增：计算平均 efs
            const auto &efs_values = efs_per_lsearch.at(l_search);
            double sum_efs = std::accumulate(efs_values.begin(), efs_values.end(), 0.0);
            double avg_efs = sum_efs / efs_values.size();

            double entry_group_search_time = 0.0;
            double entry_point_setup_time = 0.0;
            double block_authorization_time = 0.0;
            double graph_search_time = 0.0;
            double residual_time = 0.0;
            double regular_edges = 0.0;
            double free_edges = 0.0;
            double special_intra_edges = 0.0;
            double special_inter_edges = 0.0;
            double nodes_visited = 0.0;
            double total_distance_calcs = 0.0;
            double entry_point_distance_calcs = 0.0;
            double graph_search_distance_calcs = 0.0;
            size_t sample_count = 0;
            for (int repeat = 0; repeat < num_repeats; ++repeat)
            {
               for (int query_id = 0; query_id < num_queries; ++query_id)
               {
                  const auto &stats = query_stats[repeat][LsearchId][query_id];
                  entry_group_search_time += stats.get_min_super_sets_time_ms;
                  entry_point_setup_time += stats.entry_point_setup_time_ms;
                  block_authorization_time += query_block_authorization_time_ms(stats);
                  graph_search_time += query_graph_search_time_ms(stats);
                  residual_time += query_residual_time_ms(stats);
                  regular_edges += stats.regular_edges_scanned;
                  free_edges += stats.free_edges_scanned;
                  special_intra_edges += stats.special_intra_edges_scanned;
                  special_inter_edges += stats.special_inter_edges_scanned;
                  nodes_visited += stats.num_nodes_visited;
                  total_distance_calcs += stats.num_distance_calcs;
                  entry_point_distance_calcs += query_entry_point_distance_calcs(stats);
                  graph_search_distance_calcs += query_graph_search_distance_calcs(stats);
                  ++sample_count;
               }
            }
            const double divisor = sample_count > 0 ? static_cast<double>(sample_count) : 1.0;
            summary_out << l_search << "," << avg_efs << "," << avg_time << "," << avg_recall << ","
                        << entry_group_search_time / divisor << ","
                        << entry_point_setup_time / divisor << ","
                        << block_authorization_time / divisor << ","
                        << graph_search_time / divisor << ","
                        << residual_time / divisor << ","
                        << regular_edges / divisor << ","
                        << free_edges / divisor << ","
                        << special_intra_edges / divisor << ","
                        << special_inter_edges / divisor << ","
                        << (regular_edges + free_edges) / divisor << ","
                        << nodes_visited / divisor << ","
                        << total_distance_calcs / divisor << ","
                        << entry_point_distance_calcs / divisor << ","
                        << graph_search_distance_calcs / divisor << "\n";
         }
      }
      summary_out.close();
      std::cout << "性能汇总 (平均efs/耗时/召回率) 已保存到: " << summary_file_path << std::endl;

      // Per-repeat phase means let downstream analysis use warm-repeat
      // medians.  Average_Time_ms above is batch wall time and cannot be
      // decomposed by summing work performed concurrently by query threads.
      std::ofstream stage_out(result_path_prefix + "search_stage_details.csv");
      stage_out << "Repeat,Lsearch,AverageQueryTotal_ms,AverageELS_ms,"
                << "AverageEntryPointSetup_ms,AverageBlockAuthorization_ms,"
                << "AverageGraphSearch_ms,AverageResidual_ms,ClosureError_ms\n";
      for (int repeat = 0; repeat < num_repeats; ++repeat)
      {
         for (size_t LsearchId = 0; LsearchId < Lsearch_list.size(); ++LsearchId)
         {
            double total = 0.0;
            double els = 0.0;
            double entry = 0.0;
            double authorization = 0.0;
            double graph = 0.0;
            double residual = 0.0;
            for (ANNS::IdxType query_id = 0; query_id < num_queries; ++query_id)
            {
               const auto &stats = query_stats[repeat][LsearchId][query_id];
               total += stats.time_ms;
               els += stats.get_min_super_sets_time_ms;
               entry += stats.entry_point_setup_time_ms;
               authorization += query_block_authorization_time_ms(stats);
               graph += query_graph_search_time_ms(stats);
               residual += query_residual_time_ms(stats);
            }
            const double divisor = num_queries > 0 ? static_cast<double>(num_queries) : 1.0;
            const double mean_total = total / divisor;
            const double mean_els = els / divisor;
            const double mean_entry = entry / divisor;
            const double mean_authorization = authorization / divisor;
            const double mean_graph = graph / divisor;
            const double mean_residual = residual / divisor;
            const double closure = mean_total - mean_els - mean_entry -
                                   mean_authorization - mean_graph - mean_residual;
            stage_out << repeat << "," << Lsearch_list[LsearchId] << ","
                      << mean_total << "," << mean_els << ","
                      << mean_entry << "," << mean_authorization << ","
                      << mean_graph << "," << mean_residual << ","
                      << closure << "\n";
         }
      }
      stage_out.close();

      // Keep work counters per repeat so mechanism analysis can report
      // medians and dispersion without parsing the much larger per-query CSV.
      std::ofstream work_out(result_path_prefix + "search_work_details.csv");
      work_out << "Repeat,Lsearch,AverageNodesVisited,"
               << "AverageRegularNodesExpanded,AverageSpecialNodesExpanded,"
               << "AverageRegularEdgesScanned,AverageSpecialEdgesScanned,"
               << "AverageSpecialIntraEdgesScanned,AverageSpecialInterEdgesScanned,"
               << "AverageTotalEdgesScanned,AverageTotalDistanceCalcs,"
               << "AverageEntryPointDistanceCalcs,AverageGraphSearchDistanceCalcs,"
               << "AverageNumEntries,AverageEntryGroupMatchedPoints\n";
      for (int repeat = 0; repeat < num_repeats; ++repeat)
      {
         for (size_t LsearchId = 0; LsearchId < Lsearch_list.size(); ++LsearchId)
         {
            double nodes = 0.0;
            double regular_nodes = 0.0;
            double special_nodes = 0.0;
            double regular_edges = 0.0;
            double special_edges = 0.0;
            double intra_edges = 0.0;
            double inter_edges = 0.0;
            double distance_calcs = 0.0;
            double entry_distance_calcs = 0.0;
            double graph_distance_calcs = 0.0;
            double entries = 0.0;
            double matched_points = 0.0;
            for (ANNS::IdxType query_id = 0; query_id < num_queries; ++query_id)
            {
               const auto &stats = query_stats[repeat][LsearchId][query_id];
               nodes += stats.num_nodes_visited;
               regular_nodes += stats.special_regular_nodes_expanded;
               special_nodes += stats.special_free_nodes_expanded;
               regular_edges += stats.regular_edges_scanned;
               special_edges += stats.free_edges_scanned;
               intra_edges += stats.special_intra_edges_scanned;
               inter_edges += stats.special_inter_edges_scanned;
               distance_calcs += stats.num_distance_calcs;
               entry_distance_calcs += query_entry_point_distance_calcs(stats);
               graph_distance_calcs += query_graph_search_distance_calcs(stats);
               entries += stats.num_entry_points;
               matched_points += stats.entry_group_matched_points;
            }
            const double divisor = num_queries > 0 ? static_cast<double>(num_queries) : 1.0;
            work_out << repeat << "," << Lsearch_list[LsearchId] << ","
                     << nodes / divisor << ","
                     << regular_nodes / divisor << ","
                     << special_nodes / divisor << ","
                     << regular_edges / divisor << ","
                     << special_edges / divisor << ","
                     << intra_edges / divisor << ","
                     << inter_edges / divisor << ","
                     << (regular_edges + special_edges) / divisor << ","
                     << distance_calcs / divisor << ","
                     << entry_distance_calcs / divisor << ","
                     << graph_distance_calcs / divisor << ","
                     << entries / divisor << ","
                     << matched_points / divisor << "\n";
         }
      }
      work_out.close();
   }
   else
   {
      std::cerr << "错误：无法打开文件 " << summary_file_path << " 进行写入" << std::endl;
   }

   // save query details
   std::ofstream detail_out(result_path_prefix + "query_details_repeat" + std::to_string(num_repeats) + ".csv");
   detail_out << "Repeat,Lsearch,QueryID,Time_ms,EntryGroupSearchTime_ms,EntryPointSetupTime_ms,"
              << "BlockAuthorizationTime_ms,GraphSearchTime_ms,ResidualTime_ms,"
              << "RegularEdgesScanned,FreeEdgesScanned,"
              << "TotalEdgesScanned,TotalDistanceCalcs,EntryPointDistanceCalcs,"
              << "GraphSearchDistanceCalcs,core_search_time_ms,ELS_time_ms,OtherT_ms,Recall,"
              << "DistCalcs,NumNodeVisited,QuerySize,CandSize,NumEntries,EntryGroupMatchedPoints,"
              << "SpecialEntryFreePoints,SpecialEntryRegularPoints,"
              << "SpecialGroupEntryFreePoints,SpecialGroupEntryRegularPoints,"
              << "SpecialTriePivotPostings,"
              << "SpecialTrieMatchingPivots,SpecialTrieUpwardNodes,SpecialTrieDownwardNodes,"
              << "SpecialTrieBranchesPruned,SpecialTrieTerminalCandidates,"
              << "SpecialTrieTerminalDescendantsPruned,"
              << "SpecialTrieFinalEntries,SpecialTrieFinalBlockEntries,"
              << "SpecialTrieTime_ms,"
              << "EntryGroupCount,SpecialCoveredBlockCount,"
              << "SpecialFreeBlockCount,SpecialFreeBlockFrontierCount,"
              << "SpecialQueryMiddleBlockCount,SpecialQueryUpperBlockCount,"
              << "SpecialQueryUpperCoveredPoints,SpecialQueryUpperEnabled,"
              << "SpecialBlockSeedPoints,SpecialBlockSeedPointsRetained,"
              << "SpecialEntryBlocks,SpecialRetainedEntryBlocks,"
              << "SpecialBlocksSearched,SpecialMiddleBlocksSearched,SpecialUpperBlocksSearched,"
              << "SpecialBlockSearchUsed,SpecialRegularNodesExpanded,"
              << "SpecialFreeNodesExpanded,SpecialMiddleNodesExpanded,SpecialUpperNodesExpanded,"
              << "SpecialRegularEdgesScanned,SpecialFreeEdgesScanned,"
              << "SpecialIntraEdgesScanned,SpecialInterEdgesScanned,"
              << "SpecialMiddleEdgesScanned,SpecialUpperEdgesScanned,"
              << "SpecialRegularEdgeRatio,SpecialFreeEdgeRatio,"
              << "SpecialInterEdgesCoverageRejected,"
              << "SpecialCoverTime_ms,SpecialEntryTime_ms,"
              << "SpecialEdgesTime_ms,SpecialRegularEdgesTime_ms,SpecialResultTime_ms,"
              << "SpecialFreeDistanceCalcs,SpecialRegularDistanceCalcs,"
              << "SpecialEdgesAccepted,SpecialRegularEdgesAccepted,"
              << "SpecialFreeCandidatesInserted,SpecialRegularCandidatesInserted,"
              << "SpecialQueueInsertAttempts,"
              << "SpecialQueueBoundRejections,SpecialQueueInsertions,"
              << "SpecialQueueShiftedCandidates,SpecialFreeNodeCapSkipped,"
              << "SpecialFreeEdgesCapSkipped,SpecialFreeInterEdgesCapSkipped\n";
   for (int repeat = 0; repeat < num_repeats; repeat++)
   {
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         for (int i = 0; i < num_queries; ++i)
         {
            const auto &stats = query_stats[repeat][LsearchId][i];
            const double total_special_search_edges =
                static_cast<double>(stats.special_regular_edges_scanned) +
                static_cast<double>(stats.special_edges_scanned);
            const double regular_edge_ratio = total_special_search_edges > 0.0
                                                  ? stats.special_regular_edges_scanned /
                                                        total_special_search_edges
                                                  : 0.0;
            const double free_edge_ratio = total_special_search_edges > 0.0
                                               ? stats.special_edges_scanned /
                                                     total_special_search_edges
                                               : 0.0;
            const size_t graph_search_distance_calcs =
                query_graph_search_distance_calcs(stats);
            detail_out << repeat << ","
                       << Lsearch_list[LsearchId] << ","
                       << i << ","
                       << stats.time_ms << ","
                       << stats.get_min_super_sets_time_ms << ","
                       << stats.entry_point_setup_time_ms << ","
                       << query_block_authorization_time_ms(stats) << ","
                       << query_graph_search_time_ms(stats) << ","
                       << query_residual_time_ms(stats) << ","
                       << stats.regular_edges_scanned << ","
                       << stats.free_edges_scanned << ","
                       << (stats.regular_edges_scanned + stats.free_edges_scanned) << ","
                       << stats.num_distance_calcs << ","
                       << query_entry_point_distance_calcs(stats) << ","
                       << graph_search_distance_calcs << ","
                       << stats.core_search_time_ms << ","
                       << stats.get_min_super_sets_time_ms << ","
                       << query_residual_time_ms(stats) << ","
                       << stats.recall << ","
                       << stats.num_distance_calcs << ","
                       << stats.num_nodes_visited << ","
                       << stats.query_length << ","
                       << stats.candidate_set_size << ","
                       << stats.num_entry_points << ","
                       << stats.entry_group_matched_points << ","
                       << stats.special_entry_free_points << ","
                       << stats.special_entry_regular_points << ","
                       << stats.special_group_entry_free_points << ","
                       << stats.special_group_entry_regular_points << ","
                       << stats.special_trie_pivot_postings << ","
                       << stats.special_trie_matching_pivots << ","
                       << stats.special_trie_upward_nodes_visited << ","
                       << stats.special_trie_downward_nodes_visited << ","
                       << stats.special_trie_branches_pruned << ","
                       << stats.special_trie_terminal_candidates << ","
                       << stats.special_trie_terminal_descendants_pruned << ","
                       << stats.special_trie_final_entries << ","
                       << stats.special_trie_final_block_entries << ","
                       << stats.special_trie_time_ms << ","
                       << stats.num_entry_points << ","
                       << stats.special_query_block_count << ","
                       << stats.special_free_block_count << ","
                       << stats.special_free_block_frontier_count << ","
                       << stats.special_query_middle_block_count << ","
                       << stats.special_query_upper_block_count << ","
                       << stats.special_query_upper_covered_points << ","
                       << (stats.special_query_upper_enabled ? 1 : 0) << ","
                       << stats.special_block_seed_points << ","
                       << stats.special_block_seed_points_retained << ","
                       << stats.special_entry_blocks << ","
                       << stats.special_retained_entry_blocks << ","
                       << stats.special_blocks_searched << ","
                       << stats.special_middle_blocks_searched << ","
                       << stats.special_upper_blocks_searched << ","
                       << (stats.special_blocks_searched > 0 ? 1 : 0) << ","
                       << stats.special_regular_nodes_expanded << ","
                       << stats.special_free_nodes_expanded << ","
                       << stats.special_middle_nodes_expanded << ","
                       << stats.special_upper_nodes_expanded << ","
                       << stats.special_regular_edges_scanned << ","
                       << stats.special_edges_scanned << ","
                       << stats.special_intra_edges_scanned << ","
                       << stats.special_inter_edges_scanned << ","
                       << stats.special_middle_edges_scanned << ","
                       << stats.special_upper_edges_scanned << ","
                       << regular_edge_ratio << ","
                       << free_edge_ratio << ","
                       << stats.special_inter_edges_coverage_rejected << ","
                       << stats.special_cover_time_ms << ","
                       << stats.special_entry_time_ms << ","
                       << stats.special_special_edges_time_ms << ","
                       << stats.special_regular_edges_time_ms << ","
                       << stats.special_result_time_ms << ","
                       << stats.special_free_distance_calcs << ","
                       << stats.special_regular_distance_calcs << ","
                       << stats.special_edges_accepted << ","
                       << stats.special_regular_edges_accepted << ","
                       << stats.special_free_candidates_inserted << ","
                       << stats.special_regular_candidates_inserted << ","
                       << stats.special_queue_insert_attempts << ","
                       << stats.special_queue_bound_rejections << ","
                       << stats.special_queue_insertions << ","
                       << stats.special_queue_shifted_candidates << ","
                       << stats.special_free_node_cap_skipped << ","
                       << stats.special_free_edges_cap_skipped << ","
                       << stats.special_free_inter_edges_cap_skipped << "\n";
         }
      }
   }


   detail_out.close();
   std::cout << "- all done" << std::endl;
   return 0;
}
