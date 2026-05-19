#include <omp.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <vector>
#include <queue>
#include <stack>
#include <bitset>

#include <random>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <iomanip>
#include <atomic> // 引入原子操作头文件
#include <ThreadPool.h>
#include <shared_mutex>

#include "utils.h"
#include "vamana/vamana.h"
#include "include/uni_nav_graph.h"
#include "fixed_pool_graph.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

#include <mutex>
#include <cstdarg>
#include <cstdlib>
#include <string>
#include <sstream>
#include <stdexcept>


namespace fs = boost::filesystem;
using BitsetType = boost::dynamic_bitset<>;

std::shared_timed_mutex lock_m;

// 文件格式常量
const std::string FVEC_EXT = ".fvecs";
const std::string TXT_EXT = ".txt";
const size_t FVEC_HEADER_SIZE = sizeof(uint32_t);
struct QueryTask
{
   ANNS::IdxType vec_id;         // 向量ID
   std::vector<uint32_t> labels; // 查询标签集
};

struct TreeInfo
{
   ANNS::IdxType root_group_id;
   ANNS::LabelType root_label_id; // 存储概念上的根标签ID
   size_t coverage_count;
   // 用于按覆盖率降序排序
   bool operator<(const TreeInfo &other) const
   {
      return coverage_count > other.coverage_count;
   }
};



namespace {
   
   //my add
   static constexpr const char* kProfLogDir = "/home/graphdb/Codes/FilterVectorResultsCUDA/prof";
   static constexpr const char* kProfLogFile = "ung_prof.log";

   // 日志文件路径（支持用环境变量 UNG_PROF_LOG 覆盖；否则落当前目录）
   inline const std::string& prof_log_path() {
      static std::string path = []{
          namespace fs = std::filesystem;
          std::error_code ec;
          fs::create_directories(kProfLogDir, ec);
  
          // 简单时间戳：YYYYMMDD_HHMMSS
          auto now = std::chrono::system_clock::now();
          std::time_t t = std::chrono::system_clock::to_time_t(now);
          std::tm tm{};
          localtime_r(&t, &tm);
          char ts[32];
          std::snprintf(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d",
              tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  
          // 文件名：ung_prof_时间戳_进程号.log
          char fname[128];
          std::snprintf(fname, sizeof(fname), "ung_prof_%s_%d.log", ts, (int)getpid());
          return (fs::path(kProfLogDir) / fname).string();
      }();
      return path;
  }
   
   static std::mutex g_prof_mtx;
   
   // 线程安全追加一行
   inline void prof_log_line(const std::string& line) {
       std::lock_guard<std::mutex> lk(g_prof_mtx);
       std::ofstream ofs(prof_log_path(), std::ios::app);
       ofs << line << '\n';
   }
   
   // printf 风格
   inline void prof_logf(const char* fmt, ...) {
       char buf[1024];
       va_list ap; va_start(ap, fmt);
       vsnprintf(buf, sizeof(buf), fmt, ap);
       va_end(ap);
       prof_log_line(buf);
   }

   inline int read_env_int_clamped(const char *key, int fallback, int min_v, int max_v)
   {
      const char *s = std::getenv(key);
      if (!s || !*s)
         return fallback;
      char *end = nullptr;
      long v = std::strtol(s, &end, 10);
      if (end == s || *end != '\0')
         return fallback;
      if (v < (long)min_v)
         v = (long)min_v;
      if (v > (long)max_v)
         v = (long)max_v;
      return (int)v;
   }
   
   // RAII 计时器：离开作用域即落盘
   struct ScopedTimerMs {
       const char* tag;
       std::chrono::high_resolution_clock::time_point t0;
       double* acc; // 可选：把耗时累计到某个变量
       explicit ScopedTimerMs(const char* t, double* accumulate = nullptr)
           : tag(t), t0(std::chrono::high_resolution_clock::now()), acc(accumulate) {}
       ~ScopedTimerMs() {
           using namespace std::chrono;
           double ms = duration<double, std::milli>(high_resolution_clock::now() - t0).count();
           if (acc) *acc += ms;
           prof_logf("[PROF] %s %.3f", tag, ms);
       }
   };
} // anonymous namespace

namespace ANNS
{

   void UniNavGraph::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                           std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                           IdxType max_degree, IdxType Lbuild, float alpha, std::string dataset,
                           ANNS::AcornInUng new_cross_edge)
   {
      auto all_start_time = std::chrono::high_resolution_clock::now();
      _base_storage = base_storage;
      _num_points = base_storage->get_num_points();
      _distance_handler = distance_handler;
      std::cout << "- Scenario: " << scenario << std::endl;

      _dataset = dataset;

      // index parameters
      _index_name = index_name;
      _num_cross_edges = num_cross_edges;
      _max_degree = max_degree;
      _Lbuild = Lbuild;
      _alpha = alpha;
      _num_threads = num_threads;
      _scenario = scenario;

      std::cout << "Dividing groups and building the trie tree index ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      build_trie_and_divide_groups();
      _graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      _global_graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      std::cout << "begin prepare_group_storages_graphs" << std::endl;
      prepare_group_storages_graphs();
      _label_processing_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << _label_processing_time << " ms" << std::endl;

      // build graph index for each group
      build_graph_for_all_groups();
      // build_global_vamana_graph();
      build_vector_and_attr_graph(); // fxy_add

      // for label equality scenario, there is no need for label navigating graph and cross-group edges
      if (_scenario == "equality")
      {
         add_offset_for_uni_nav_graph();
      }
      else
      {

         // build the label navigating graph
         build_label_nav_graph();
         get_descendants_info(); // fxy_add
         {
            std::cout << "\n--- [分析] 正在统计 LNG 边数 ---" << std::endl;
            uint64_t total_lng_edges = 0;
            for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)// 遍历所有组，累加它们的“出度”（即子节点/超集数量）
            {
               total_lng_edges += _label_nav_graph->out_neighbors[group_id].size();
            }
            std::cout << "  - 组 (Nodes) 总数: " << _num_groups << std::endl;
            std::cout << "  - LNG 边 (Edges) 总数: " << total_lng_edges << std::endl;
            if (_num_groups > 0) {
                 std::cout << "  - 平均出度 (边/组): " << static_cast<double>(total_lng_edges) / _num_groups << std::endl;
            }
            std::cout << "--- [分析] LNG 边数统计完毕 ---\n" << std::endl;
         }

         // calculate the coverage ratio
         cal_f_coverage_ratio(); // fxy_add

         // initialize_lng_descendants_coverage_bitsets();
         auto roaring_start_time = std::chrono::high_resolution_clock::now();
         initialize_roaring_bitsets();
         _build_roaring_bitsets_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - roaring_start_time).count();

         // precompute the node depths in the label navigating graph,used in hard sandwitch
         // _precompute_lng_node_depths();

         std::cout << "new_cross_edge.ung_and_acorn: " << new_cross_edge.ung_and_acorn << std::endl;
         if (!new_cross_edge.ung_and_acorn)
            build_cross_group_edges();
         else
         {
            _num_cross_edges = _num_cross_edges / 2;
            finalize_intra_group_graphs(); // fxy_add:从 build_cross_group_edges中拎出转换id新旧转换的部分

            add_new_distance_oriented_edges(
                dataset,
                num_threads,
                new_cross_edge);
         }
      }

      // index time
      _index_time = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - all_start_time)
                        .count();
   }

   void UniNavGraph::build_trie_and_divide_groups()
   {
      // create groups for base label sets
      IdxType new_group_id = 1;
      for (auto vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         auto group_id = _trie_index.insert(label_set, new_group_id);

         // deal with new label setinver
         if (group_id + 1 > _group_id_to_vec_ids.size())
         {
            _group_id_to_vec_ids.resize(group_id + 1);
            _group_id_to_label_set.resize(group_id + 1);
            _group_id_to_label_set[group_id] = label_set;
         }
         _group_id_to_vec_ids[group_id].emplace_back(vec_id);
      }

      // logs
      _num_groups = new_group_id - 1;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
   }

   void UniNavGraph::get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                        bool avoid_self, bool need_containment)
   {
      min_super_set_ids.clear();

      // obtain the candidates
      // 复用线程本地缓存，避免每次调用反复分配/释放候选容器。
      thread_local std::vector<std::shared_ptr<TrieNode>> candidates;
      candidates.clear();
      _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment); // 搜索候选超集

      // ================= [DEBUG] =================
      // 如果候选集数量超过1000000，说明这个组非常容易导致卡死
      if (candidates.size() > 1000000) 
      {
         #pragma omp critical
         {
            std::cout << "[WARNING] Performance Hazard! Group Label Size: " << query_label_set.size()
                        << " | Candidates Found: " << candidates.size() 
                        << " (This will be slow!)" << std::endl;
            if (!query_label_set.empty()) {
                  std::cout << "   -> Labels: " << query_label_set[0] << " ... " << std::endl;
            }
         }
      }
      // ===================================================

      // special cases
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      thread_local std::vector<TrieNode*> candidate_nodes;
      candidate_nodes.clear();
      candidate_nodes.reserve(candidates.size());
      LabelType min_size = candidates[0]->label_set_size;
      LabelType max_size = min_size;
      for (const auto &sp : candidates)
      {
         TrieNode *node = sp.get();
         candidate_nodes.push_back(node);
         min_size = std::min(min_size, node->label_set_size);
         max_size = std::max(max_size, node->label_set_size);
      }

      thread_local std::vector<const std::vector<LabelType>*> min_label_sets;
      min_label_sets.clear();
      min_super_set_ids.reserve(candidate_nodes.size());

      auto try_add_candidate = [&](const TrieNode *candidate) {
         const auto cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         // check whether contains existing minimum super sets (label ids are in ascending order)
         if (candidate->label_set_size > min_size)
         {
            for (const auto *min_label_set : min_label_sets)
            {
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set->begin(), min_label_set->end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         if (is_min)
         {
            min_super_set_ids.emplace_back(cur_group_id);
            min_label_sets.emplace_back(&cur_label_set);
         }
      };

      // 当 label_set_size 跨度较小时，使用桶化遍历替代 O(n log n) 排序。
      static constexpr LabelType kBucketSpanThreshold = 256;
      if (max_size >= min_size && (max_size - min_size) <= kBucketSpanThreshold)
      {
         const size_t bucket_span = static_cast<size_t>(max_size - min_size + 1);
         thread_local std::vector<std::vector<TrieNode*>> size_buckets;
         if (size_buckets.size() < bucket_span)
            size_buckets.resize(bucket_span);
         for (size_t i = 0; i < bucket_span; ++i)
            size_buckets[i].clear();

         for (TrieNode *node : candidate_nodes)
            size_buckets[static_cast<size_t>(node->label_set_size - min_size)].push_back(node);

         for (size_t i = 0; i < bucket_span; ++i)
            for (const TrieNode *candidate : size_buckets[i])
               try_add_candidate(candidate);
      }
      else
      {
         // 跨度较大时回退原排序路径，保证稳定性。
         std::sort(candidate_nodes.begin(), candidate_nodes.end(),
                   [](const TrieNode *a, const TrieNode *b)
                   {
                      return a->label_set_size < b->label_set_size;
                   });

         for (const TrieNode *candidate : candidate_nodes)
            try_add_candidate(candidate);
      }
   }

   void UniNavGraph::get_min_super_sets_debug(const std::vector<LabelType> &query_label_set,
                                              std::vector<IdxType> &min_super_set_ids,
                                              bool avoid_self, bool need_containment,
                                              std::atomic<int> &print_counter, bool is_new_trie_method, bool is_rec_more_start, QueryStats &stats,
                                              bool skip_group_id_check)
   {
#if ENABLE_ENTRY_DEBUG_OUTPUT
      // --- 计时器变量定义 ---
      double time_trie_lookup = 0.0;
      double time_sorting = 0.0;
      double time_filtering_loop = 0.0;
      auto function_start_time = std::chrono::high_resolution_clock::now();
#endif

      min_super_set_ids.clear();

      // --- 1. 测量Trie查找候选者的时间 ---
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_trie = std::chrono::high_resolution_clock::now();
#endif

      // // 对应伪代码 L1-L8: 增加快速路径，首先尝试精确匹配
      // auto exact_match_node = _trie_index.find_exact_match(query_label_set);
      // if (exact_match_node != nullptr && exact_match_node->group_id > 0)
      // {
      //    min_super_set_ids.emplace_back(exact_match_node->group_id);
      //    return;
      // }

      std::vector<std::shared_ptr<TrieNode>> candidates;

      if (is_new_trie_method)
      {
         // --- 调用新方法 (Recursive) ---
         TrieSearchMetricsRecursive trie_metrics_m2 = {};
         if (is_rec_more_start)
            _trie_index.get_super_set_entrances_new_more_sp_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);
         else
            _trie_index.get_super_set_entrances_new_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);
         stats.recursive_calls = trie_metrics_m2.recursive_calls;
         stats.pruning_events = trie_metrics_m2.pruning_events;
         stats.trie_nodes_traversed = trie_metrics_m2.recursive_calls + trie_metrics_m2.nodes_processed_in_bfs;
         if (stats.recursive_calls > 0)
            stats.pruning_efficiency = static_cast<float>(stats.pruning_events) / stats.recursive_calls;
         else
            stats.pruning_efficiency = 0.0f;
      }
      else
      {
         // --- 调用旧方法 (Shortcut) ---
         TrieMethod1Metrics trie_metrics_m1 = {};
         _trie_index.get_super_set_entrances_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m1, skip_group_id_check);
         stats.successful_checks = trie_metrics_m1.successful_checks;
         stats.trie_nodes_traversed = trie_metrics_m1.upward_traversals + trie_metrics_m1.bfs_nodes_processed;
         stats.redundant_upward_steps = trie_metrics_m1.redundant_upward_steps;
         if (stats.candidate_set_size > 0)
            stats.shortcut_hit_ratio = static_cast<float>(stats.successful_checks) / stats.candidate_set_size;
         else
            stats.shortcut_hit_ratio = 0.0f;
      }

#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_trie = std::chrono::high_resolution_clock::now();
      time_trie_lookup = std::chrono::duration<double, std::milli>(end_trie - start_trie).count();
#endif

      // 特殊情况处理
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      // --- 2. 测量排序时间 ---
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_sort = std::chrono::high_resolution_clock::now();
#endif
      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_sort = std::chrono::high_resolution_clock::now();
      time_sorting = std::chrono::duration<double, std::milli>(end_sort - start_sort).count();
#endif

      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      // --- 3. 测量过滤循环的时间 ---
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_filter = std::chrono::high_resolution_clock::now();
#endif
      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         if (is_min)
         {
            min_super_set_ids.emplace_back(cur_group_id);
         }
      }
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_filter = std::chrono::high_resolution_clock::now();
      time_filtering_loop = std::chrono::duration<double, std::milli>(end_filter - start_filter).count();

      auto function_total_time = std::chrono::duration<double, std::milli>(end_filter - function_start_time).count();
#endif
   }

   // 为了方便排序，定义一个结构体来存储候选组及其评分
   struct ScoredCandidate
   {
      double score;
      ANNS::IdxType group_id;

      // 重载小于运算符，方便使用 std::sort 进行降序排序
      bool operator<(const ScoredCandidate &other) const
      {
         return score > other.score; // score 越大越靠前
      }
   };

   std::vector<ANNS::IdxType> UniNavGraph::select_entry_groups(
       const std::vector<ANNS::IdxType> &minimum_entry_sets,
       ANNS::SelectionMode mode,
       size_t top_k,
       double beta,
       ANNS::IdxType true_query_group_id) const
   {
      bool verbose = false;
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph should not be null.");

      if (top_k == 0 && true_query_group_id == 0)
      {
         return minimum_entry_sets;
      }
      if (minimum_entry_sets.empty() && true_query_group_id == 0)
      {
         return {};
      }

      // --- 步骤 1: 运行BFS，并收集所有可达的候选节点 ---
      std::queue<ANNS::IdxType> q;
      std::vector<int> lng_distance(_num_groups + 1, -1);

      // === 核心优化 1: 创建一个容器，仅用于存储BFS访问过的、距离大于0的节点 ===
      std::vector<ANNS::IdxType> reachable_nodes;
      reachable_nodes.reserve(_num_groups / 10); // 预分配一些内存，避免多次重分配

      for (const auto &group_id : minimum_entry_sets)
      {
         if (group_id > 0 && group_id <= _num_groups && lng_distance[group_id] == -1)
         {
            q.push(group_id);
            lng_distance[group_id] = 0;
         }
      }

      while (!q.empty())
      {
         ANNS::IdxType current_group = q.front();
         q.pop();

         if (current_group >= _label_nav_graph->out_neighbors.size())
            continue;

         for (const auto &child_group : _label_nav_graph->out_neighbors[current_group])
         {
            if (child_group > 0 && child_group <= _num_groups && lng_distance[child_group] == -1)
            {
               lng_distance[child_group] = lng_distance[current_group] + 1;
               q.push(child_group);
               // === 核心优化 2: 将新发现的候选节点存入列表 ===
               reachable_nodes.push_back(child_group);
            }
         }
      }

      // --- 步骤 2: 仅对可达节点进行评分 ---
      std::vector<ANNS::ScoredCandidate> candidates;
      candidates.reserve(reachable_nodes.size());

      // === 核心优化 3: 直接遍历数量少得多的 reachable_nodes 列表 ===
      for (const auto group_id : reachable_nodes)
      {
         const double group_size = static_cast<double>(_group_id_to_vec_ids[group_id].size());
         double score = 0.0;
         if (mode == ANNS::SelectionMode::SizeOnly)
         {
            score = group_size;
         }
         else
         { // SizeAndDistance with new formula
            const int dist = lng_distance[group_id];
            // 新评分公式: 深度优先，大小作为次要排序依据
            score = static_cast<double>(dist) * static_cast<double>(_num_points) + group_size;
         }
         candidates.push_back({score, group_id});
      }

      // --- 步骤 3: 部分排序并选择 ---
      size_t num_to_sort = std::min(candidates.size(), top_k + 5); // +5 作为安全缓冲
      if (num_to_sort > 0)
      {
         std::partial_sort(candidates.begin(),
                           candidates.begin() + num_to_sort,
                           candidates.end());
      }

      std::vector<ANNS::IdxType> final_entry_groups = minimum_entry_sets;
      std::unordered_set<ANNS::IdxType> existing_groups(minimum_entry_sets.begin(), minimum_entry_sets.end());

      bool oracle_injected = false;
      bool oracle_existed = false;

      if (true_query_group_id > 0 && true_query_group_id <= _num_groups)
      {
         if (existing_groups.find(true_query_group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(true_query_group_id);
            existing_groups.insert(true_query_group_id);
            oracle_injected = true;
         }
         else
         {
            oracle_existed = true;
         }
      }

      for (size_t i = 0; i < num_to_sort; ++i)
      {
         const auto &candidate = candidates[i];
         if (final_entry_groups.size() >= minimum_entry_sets.size() + top_k)
         {
            break;
         }
         if (existing_groups.find(candidate.group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(candidate.group_id);
            existing_groups.insert(candidate.group_id);
         }
      }

      // --- 精简后的日志输出 ---
      if (verbose)
      {
         if (oracle_injected)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was successfully injected." << std::endl;
         }
         else if (oracle_existed && true_query_group_id > 0)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was already in the initial set." << std::endl;
         }

         size_t added_count = final_entry_groups.size() > minimum_entry_sets.size() ? final_entry_groups.size() - minimum_entry_sets.size() : 0;
         if (oracle_injected)
            added_count--;

         std::cout << "[Info] Final selected entry groups count: " << final_entry_groups.size()
                   << " (Initial: " << minimum_entry_sets.size()
                   << ", Heuristically Added: " << added_count << ")" << std::endl;
      }

      return final_entry_groups;
   }

   void UniNavGraph::prepare_group_storages_graphs()
   {
      _new_vec_id_to_group_id.resize(_num_points);

      // reorder the vectors
      _group_id_to_range.resize(_num_groups + 1);
      _new_to_old_vec_ids.resize(_num_points);
      IdxType new_vec_id = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         _group_id_to_range[group_id].first = new_vec_id;
         for (auto old_vec_id : _group_id_to_vec_ids[group_id])
         {
            _new_to_old_vec_ids[new_vec_id] = old_vec_id;
            _new_vec_id_to_group_id[new_vec_id] = group_id;
            ++new_vec_id;
         }
         _group_id_to_range[group_id].second = new_vec_id;
      }

      // reorder the underlying storage
      _base_storage->reorder_data(_new_to_old_vec_ids);

      // init storage and graph for each group
      _group_storages.resize(_num_groups + 1);
      _group_graphs.resize(_num_groups + 1);
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         auto start = _group_id_to_range[group_id].first;
         auto end = _group_id_to_range[group_id].second;
         _group_storages[group_id] = create_storage(_base_storage, start, end);
         _group_graphs[group_id] = std::make_shared<Graph>(_graph, start, end);
      }
   }

   void UniNavGraph::build_graph_for_all_groups()
   {
      std::cout << "Building graph for each group ..." << std::endl;
      omp_set_num_threads(_num_threads);
      auto start_time = std::chrono::high_resolution_clock::now();

      // build group-level proximity graphs
      if (_index_name == "Vamana" || _index_name == "FixedPoolGPU")
      {
         _vamana_instances.resize(_num_groups + 1);
         _group_entry_points.resize(_num_groups + 1);

#pragma omp parallel for schedule(dynamic, 1)
         for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         {
            // if (group_id % 100 == 0)
            //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;

            // if there are less than _max_degree points in the group, just build a complete graph
            const auto &range = _group_id_to_range[group_id];
            if (range.second - range.first <= _max_degree)
            {
               build_complete_graph(_group_graphs[group_id], range.second - range.first);
               _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                      _group_graphs[group_id], 0);

               // build the vamana graph
            }
            else if (_index_name == "Vamana")
            {
               _vamana_instances[group_id] = std::make_shared<Vamana>(false);
               _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                                  _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
            }
            else
            {
               const uint32_t fixed_pool_iters = 4;
               const uint64_t fixed_pool_seed = 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(group_id);
               build_fixed_pool_graph_gpu(_group_storages[group_id], _distance_handler,
                                          _group_graphs[group_id], _max_degree, _Lbuild,
                                          _alpha, fixed_pool_iters, fixed_pool_seed);
               _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                      _group_graphs[group_id], 0);
            }

            // set entry point
            _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         }

         // if none of the above
      }
      else
      {
         std::cerr << "Error: invalid index name " << _index_name << std::endl;
         exit(-1);
      }
      _build_graph_time = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
   }

   // fxy_add：构建全局Vamana图
   void UniNavGraph::build_global_vamana_graph()
   {
      std::cout << "Building global Vamana graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _global_vamana = std::make_shared<Vamana>(false);

      _global_vamana->build(_base_storage, _distance_handler, _global_graph, _max_degree, _Lbuild, _alpha, 1);

      auto build_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      _global_vamana_entry_point = _global_vamana->get_entry_point();

      std::cout << "- Global Vamana graph built in " << build_time << " ms" << std::endl;
   }
   //=====================================begin 数据预处理：构建向量-属性二分图=========================================
   // fxy_add: 构建向量-属性二分图
   void UniNavGraph::build_vector_and_attr_graph()
   {
      std::cout << "Building vector-attribute bipartite graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // 初始化属性到ID的映射和反向映射
      _attr_to_id.clear();
      _id_to_attr.clear();
      _vector_attr_graph.clear();

      // 第一遍：收集所有唯一属性并分配ID
      AtrType attr_id = 0;
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            if (_attr_to_id.find(label) == _attr_to_id.end())
            {
               _attr_to_id[label] = attr_id;
               _id_to_attr[attr_id] = label;
               attr_id++;
            }
         }
      }

      // 初始化图结构（向量节点 + 属性节点）
      _vector_attr_graph.resize(_num_points + static_cast<size_t>(attr_id));

      // 第二遍：构建图结构
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            AtrType a_id = _attr_to_id[label];

            // 添加双向边
            // 向量节点ID范围: [0, _num_points-1]
            // 属性节点ID范围: [_num_points, _num_points+attr_id-1]
            _vector_attr_graph[vec_id].push_back(_num_points + static_cast<IdxType>(a_id));
            _vector_attr_graph[_num_points + static_cast<IdxType>(a_id)].push_back(vec_id);
         }
      }

      // 统计信息
      _num_attributes = attr_id;
      _build_vector_attr_graph_time = std::chrono::duration<double, std::milli>(
                                          std::chrono::high_resolution_clock::now() - start_time)
                                          .count();
      std::cout << "- Finish in " << _build_vector_attr_graph_time << " ms" << std::endl;
      std::cout << "- Total edges: " << count_graph_edges() << std::endl;

      // 可选：保存图结构供调试
      // save_bipartite_graph_info();
   }

   // fxy_add: 计算向量-属性二分图的边数
   size_t UniNavGraph::count_graph_edges() const
   {
      size_t total_edges = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         total_edges += neighbors.size();
      }
      return total_edges / 2; // 因为是双向边，实际边数是总数的一半
   }

   // fxy_add: 保存二分图信息到txt文件调试用
   void UniNavGraph::save_bipartite_graph_info() const
   {
      std::ofstream outfile("bipartite_graph_info.txt");
      if (!outfile.is_open())
      {
         std::cerr << "Warning: Could not open file to save bipartite graph info" << std::endl;
         return;
      }

      outfile << "Bipartite Graph Information\n";
      outfile << "==========================\n";
      outfile << "Total vectors: " << _num_points << "\n";
      outfile << "Total attributes: " << _num_attributes << "\n";
      outfile << "Total edges: " << count_graph_edges() << "\n\n";

      // 输出属性映射
      outfile << "Attribute to ID Mapping:\n";
      for (const auto &pair : _attr_to_id)
      {
         outfile << pair.first << " -> " << static_cast<IdxType>(pair.second) << "\n";
      }
      outfile << "\n";

      // 输出部分图结构示例
      outfile << "Sample Graph Connections (first 10 vectors and attributes):\n";
      outfile << "Vector connections:\n";
      for (IdxType i = 0; i < std::min(_num_points, static_cast<IdxType>(10)); ++i)
      {
         outfile << "Vector " << i << " connects to attributes: ";
         for (auto a_node : _vector_attr_graph[i])
         {
            AtrType attr_id = static_cast<AtrType>(a_node - _num_points);
            outfile << _id_to_attr.at(attr_id) << " ";
         }
         outfile << "\n";
      }

      outfile << "\nAttribute connections:\n";
      for (AtrType i = 0; i < std::min(_num_attributes, static_cast<AtrType>(5)); ++i)
      {
         outfile << "Attribute " << _id_to_attr.at(i) << " connects to vectors: ";
         for (auto v_node : _vector_attr_graph[_num_points + static_cast<IdxType>(i)])
         {
            outfile << v_node << " ";
         }
         outfile << "\n";
      }

      outfile.close();
      std::cout << "- Bipartite graph info saved to bipartite_graph_info.txt" << std::endl;
   }

   uint32_t UniNavGraph::compute_checksum() const
   {
      // 简单的校验和计算示例
      uint32_t sum = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         for (IdxType node : neighbors)
         {
            sum ^= (node << (sum % 32));
         }
      }
      return sum;
   }

   // fxy_add: 保存二分图到文件
   void UniNavGraph::save_bipartite_graph(const std::string &filename)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }

      // 1. 写入文件头标识和版本
      const char header[8] = {'B', 'I', 'P', 'G', 'R', 'P', 'H', '1'};
      out.write(header, 8);

      // 2. 写入基本元数据
      out.write(reinterpret_cast<const char *>(&_num_points), sizeof(IdxType));
      out.write(reinterpret_cast<const char *>(&_num_attributes), sizeof(AtrType));

      // 3. 写入属性映射表
      // 3.1 先写入条目数量
      uint64_t map_size = _attr_to_id.size();
      out.write(reinterpret_cast<const char *>(&map_size), sizeof(uint64_t));

      // 3.2 写入每个映射条目（LabelType是uint16_t，直接存储）
      for (const auto &[label, id] : _attr_to_id)
      {
         out.write(reinterpret_cast<const char *>(&label), sizeof(LabelType));
         out.write(reinterpret_cast<const char *>(&id), sizeof(AtrType));
      }

      // 4. 写入邻接表数据
      // 4.1 先写入节点总数
      uint64_t total_nodes = _vector_attr_graph.size();
      out.write(reinterpret_cast<const char *>(&total_nodes), sizeof(uint64_t));

      // 4.2 写入每个节点的邻居列表
      for (const auto &neighbors : _vector_attr_graph)
      {
         // 先写入邻居数量
         uint32_t neighbor_count = neighbors.size();
         out.write(reinterpret_cast<const char *>(&neighbor_count), sizeof(uint32_t));

         // 写入邻居ID列表
         if (!neighbors.empty())
         {
            out.write(reinterpret_cast<const char *>(neighbors.data()),
                      neighbors.size() * sizeof(IdxType));
         }
      }

      // 5. 写入文件尾校验和
      uint32_t checksum = compute_checksum();
      out.write(reinterpret_cast<const char *>(&checksum), sizeof(uint32_t));

      std::cout << "Successfully saved bipartite graph to " << filename
                << " (" << out.tellp() << " bytes)" << std::endl;
   }

   // fxy_add: 读取二分图
   void UniNavGraph::load_bipartite_graph(const std::string &filename)
   {
      std::cout << "Loading bipartite graph from " << filename << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         throw std::runtime_error("Cannot open file for reading: " + filename);
      }

      // 1. 验证文件头
      char header[8];
      in.read(header, 8);
      if (std::string(header, 8) != "BIPGRPH1")
      {
         throw std::runtime_error("Invalid file format");
      }

      // 2. 读取基本元数据
      in.read(reinterpret_cast<char *>(&_num_points), sizeof(IdxType));
      in.read(reinterpret_cast<char *>(&_num_attributes), sizeof(AtrType));

      // 3. 读取属性映射表
      _attr_to_id.clear();
      _id_to_attr.clear();

      // 3.1 读取条目数量
      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(uint64_t));

      // 3.2 读取每个映射条目
      for (uint64_t i = 0; i < map_size; ++i)
      {
         LabelType label;
         AtrType id;

         in.read(reinterpret_cast<char *>(&label), sizeof(LabelType));
         in.read(reinterpret_cast<char *>(&id), sizeof(AtrType));

         _attr_to_id[label] = id;
         _id_to_attr[id] = label;
      }

      // 4. 读取邻接表数据
      _vector_attr_graph.clear();

      // 4.1 读取节点总数
      uint64_t total_nodes;
      in.read(reinterpret_cast<char *>(&total_nodes), sizeof(uint64_t));
      _vector_attr_graph.resize(total_nodes);

      // 4.2 读取每个节点的邻居列表
      for (uint64_t i = 0; i < total_nodes; ++i)
      {
         uint32_t neighbor_count;
         in.read(reinterpret_cast<char *>(&neighbor_count), sizeof(uint32_t));

         _vector_attr_graph[i].resize(neighbor_count);
         if (neighbor_count > 0)
         {
            in.read(reinterpret_cast<char *>(_vector_attr_graph[i].data()),
                    neighbor_count * sizeof(IdxType));
         }
      }

      // 5. 验证校验和
      uint32_t stored_checksum;
      in.read(reinterpret_cast<char *>(&stored_checksum), sizeof(uint32_t));

      uint32_t computed_checksum = compute_checksum();
      if (stored_checksum != computed_checksum)
      {
         throw std::runtime_error("Checksum verification failed");
      }

      std::cout << "- Loaded bipartite graph with " << _num_points << " vectors and "
                << _num_attributes << " attributes in "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - start_time)
                       .count()
                << " ms" << std::endl;
   }

   // fxy_add: 比较两个向量-属性二分图
   bool UniNavGraph::compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2)
   {
      // 1. 验证基本属性
      if (g1._num_points != g2._num_points)
      {
         std::cerr << "Mismatch in _num_points: "
                   << g1._num_points << " vs " << g2._num_points << std::endl;
         return false;
      }

      if (g1._num_attributes != g2._num_attributes)
      {
         std::cerr << "Mismatch in _num_attributes: "
                   << g1._num_attributes << " vs " << g2._num_attributes << std::endl;
         return false;
      }

      // 2. 验证属性映射
      if (g1._attr_to_id.size() != g2._attr_to_id.size())
      {
         std::cerr << "Mismatch in _attr_to_id size" << std::endl;
         return false;
      }

      for (const auto &[label, id] : g1._attr_to_id)
      {
         auto it = g2._attr_to_id.find(label);
         if (it == g2._attr_to_id.end())
         {
            std::cerr << "Label " << label << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != id)
         {
            std::cerr << "Mismatch ID for label " << label
                      << ": " << id << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 3. 验证反向属性映射
      for (const auto &[id, label] : g1._id_to_attr)
      {
         auto it = g2._id_to_attr.find(id);
         if (it == g2._id_to_attr.end())
         {
            std::cerr << "ID " << id << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != label)
         {
            std::cerr << "Mismatch label for ID " << id
                      << ": " << label << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 4. 验证邻接表
      if (g1._vector_attr_graph.size() != g2._vector_attr_graph.size())
      {
         std::cerr << "Mismatch in graph size" << std::endl;
         return false;
      }

      for (size_t i = 0; i < g1._vector_attr_graph.size(); ++i)
      {
         const auto &neighbors1 = g1._vector_attr_graph[i];
         const auto &neighbors2 = g2._vector_attr_graph[i];

         if (neighbors1.size() != neighbors2.size())
         {
            std::cerr << "Mismatch in neighbors count for node " << i << std::endl;
            return false;
         }

         for (size_t j = 0; j < neighbors1.size(); ++j)
         {
            if (neighbors1[j] != neighbors2[j])
            {
               std::cerr << "Mismatch in neighbor " << j << " for node " << i
                         << ": " << neighbors1[j] << " vs " << neighbors2[j] << std::endl;
               return false;
            }
         }
      }

      std::cout << "Graphs are identical!" << std::endl;

      return true;
   }
   //=====================================end 数据预处理：构建向量-属性二分图=========================================

   //=====================================begein 查询过程：计算bitmap=========================================

   // fxy_add: 构建bitmap
   std::pair<std::bitset<10000001>, double> UniNavGraph::compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const
   {
      // 1. 初始化全true的bitmap（表示开始时所有点都满足条件）
      std::bitset<10000001> bitmap;
      bitmap.set(); // 设置所有位为1
      double per_query_bitmap_time = 0.0;

      // 2. 处理每个查询属性
      for (LabelType attr_label : query_attributes)
      {
         //  2.1 查找属性ID
         auto it = _attr_to_id.find(attr_label);
         if (it == _attr_to_id.end())
         {
            // 属性不存在，没有任何点能满足所有条件
            return {std::bitset<10000001>(), 0.0};
         }

         // 2.2 获取属性节点ID
         AtrType attr_id = it->second;
         IdxType attr_node_id = _num_points + static_cast<IdxType>(attr_id);

         // 2.3 创建临时bitmap记录当前属性的满足情况
         std::bitset<10000001> temp_bitmap;
         auto start_time = std::chrono::high_resolution_clock::now();
         for (IdxType vec_id : _vector_attr_graph[attr_node_id])
         {
            temp_bitmap.set(vec_id); // 使用set()方法设置对应的位为1
         }

         // 2.4 与主bitmap进行AND操作
         bitmap &= temp_bitmap; // 使用&=操作符进行位与操作

         per_query_bitmap_time += std::chrono::duration<double, std::milli>(
                                      std::chrono::high_resolution_clock::now() - start_time)
                                      .count();
      }

      return {bitmap, per_query_bitmap_time};
   }

   // fxy_add: 通过分组ID计算bitmap
   roaring::Roaring UniNavGraph::compute_bitmap_from_groups(const std::vector<IdxType> &group_ids) const
   {
      roaring::Roaring final_bitmap;
      for (IdxType group_id : group_ids)
      {
         // 确保group_id有效
         if (group_id > 0 && group_id <= _num_groups)
         {
            // 对每个分组的覆盖集（_covered_sets_rb）执行并集操作
            final_bitmap |= _covered_sets_rb[group_id];
         }
      }
      return final_bitmap;
   }

   // fxy_add: 批量计算UNG过滤的bitmaps
   std::vector<roaring::Roaring> UniNavGraph::batch_compute_ung_bitmaps(
       const ANNS::UniNavGraph &index,
       const std::shared_ptr<ANNS::IStorage> &query_storage,
       uint32_t num_threads,
       bool is_new_trie_method,
       bool is_rec_more_start)
   {
      auto num_queries = query_storage->get_num_points();
      std::vector<roaring::Roaring> all_bitmaps(num_queries);

// omp_set_num_threads(num_threads);
#pragma omp parallel
      {
         // 为每个线程创建一个独立的原子计数器，避免在循环中重复创建. static 确保了多线程环境下只有一个实例
         static std::atomic<int> trie_debug_print_counter{0};

#pragma omp for
         for (int id = 0; id < num_queries; ++id)
         {
            // 1. 获取当前查询的标签
            const auto &query_labels = query_storage->get_label_set(id);

            // 2. 获取最小超集分组
            std::vector<ANNS::IdxType> entry_group_ids;

            // 为 get_min_super_sets_debug 创建一个临时的、仅在此作用域有效的 stats 对象
            ANNS::QueryStats dummy_stats;

            const_cast<ANNS::UniNavGraph &>(index).get_min_super_sets_debug(
                query_labels,
                entry_group_ids,
                false, true, // avoid_self=false, need_containment=true 是通常的默认值
                trie_debug_print_counter,
                is_new_trie_method,
                is_rec_more_start,
                dummy_stats,false);

            // 3. 利用分组ID高效计算bitmap
            all_bitmaps[id] = index.compute_bitmap_from_groups(entry_group_ids);
         }
      }
      return all_bitmaps;
   }

   //====================================end 查询过程：计算bitmap=========================================
   void UniNavGraph::build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points)
   {
      for (auto i = 0; i < num_points; ++i)
         for (auto j = 0; j < num_points; ++j)
            if (i != j)
               graph->neighbors[i].emplace_back(j);
   }

   //=====================================begin LNG中每个f覆盖率计算=========================================
   // fxy_add: 递归打印孩子节点及其覆盖率
   void print_children_recursive(const std::shared_ptr<ANNS::LabelNavGraph> graph, IdxType group_id, std::ofstream &outfile, int indent_level)
   {
      // 根据缩进级别生成前缀空格
      std::string indent(indent_level * 4, ' '); // 每一层缩进4个空格

      // 打印当前节点
      outfile << "\n"
              << indent << "[" << group_id << "] (" << graph->coverage_ratio[group_id] << ")";

      // 如果有子节点，递归打印子节点
      if (!graph->out_neighbors[group_id].empty())
      {
         for (size_t i = 0; i < graph->out_neighbors[group_id].size(); ++i)
         {
            auto child_group_id = graph->out_neighbors[group_id][i];
            print_children_recursive(graph, child_group_id, outfile, indent_level + 1); // 增加缩进级别
         }
      }
   }

   // fxy_add: 输出覆盖率及孩子节点,调用print_children_recursive
   void output_coverage_ratio(const std::shared_ptr<ANNS::LabelNavGraph> _label_nav_graph, IdxType _num_groups, std::ofstream &outfile)
   {
      outfile << "Coverage Ratio for each group\n";
      outfile << "=====================================\n";
      outfile << "Format: [GroupID] -> CoverageRatio \n";
      outfile << "-> [child_GroupID (CoverageRatio)]\n\n";

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 打印当前组的覆盖率
         outfile << "[" << group_id << "] -> " << _label_nav_graph->coverage_ratio[group_id];

         // 打印孩子节点
         if (!_label_nav_graph->out_neighbors[group_id].empty())
         {
            outfile << " ->";
            for (size_t i = 0; i < _label_nav_graph->out_neighbors[group_id].size(); ++i)
            {
               auto child_group_id = _label_nav_graph->out_neighbors[group_id][i];
               print_children_recursive(_label_nav_graph, child_group_id, outfile, 1); // 第一层子节点缩进为1
            }
         }
         outfile << "\n";
      }
   }

   // fxy_add：计算每个label set的向量覆盖比率
   void UniNavGraph::cal_f_coverage_ratio()
   {
      std::cout << "Calculating coverage ratio..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      int coverage_threads = _num_threads;
      if (const char *env = std::getenv("UNG_COVERAGE_THREADS"))
      {
         int v = std::atoi(env);
         if (v > 0)
            coverage_threads = std::min<int>(v, _num_threads);
      }
      std::cout << "- coverage threads: " << coverage_threads << std::endl;

      // Step 0: 初始化covered_sets
      _label_nav_graph->coverage_ratio.clear();
      _label_nav_graph->covered_sets.clear();
      _label_nav_graph->covered_sets.resize(_num_groups + 1);

      // Step 1-3: 直接基于已计算的 descendants 构建覆盖集合，避免拓扑传播中的重复哈希合并。
      bool use_descendants_direct = false;

      if (const char *env = std::getenv("UNG_COVERAGE_IMPL"))
      {
         // 0=legacy(topological merge), 1=descendants_direct
         use_descendants_direct = (std::atoi(env) != 0);
      }

      if (use_descendants_direct)
      {
         std::cout << "- coverage impl: descendants_direct" << std::endl;
#pragma omp parallel for schedule(dynamic, 64) num_threads(coverage_threads)
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            auto &coverage = _label_nav_graph->covered_sets[group_id];
            coverage.clear();
            const auto &self_vec_ids = _group_id_to_vec_ids[group_id];
            const auto &descendants = _label_nav_graph->_lng_descendants[group_id];

            size_t reserve_n = self_vec_ids.size();
            for (const auto desc_group_id : descendants)
            {
               reserve_n += _group_id_to_vec_ids[desc_group_id].size();
            }

            coverage.reserve(reserve_n);
            coverage.insert(coverage.end(), self_vec_ids.begin(), self_vec_ids.end());
            for (const auto desc_group_id : descendants)
            {
               const auto &vec_ids = _group_id_to_vec_ids[desc_group_id];
               coverage.insert(coverage.end(), vec_ids.begin(), vec_ids.end());
            }
         }
      }
      else
      {
         std::cout << "- coverage impl: legacy_topological_merge" << std::endl;
         // Step 1: 初始化每个 group 的覆盖集合（并行）
#pragma omp parallel for schedule(dynamic, 256) num_threads(coverage_threads)
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &vec_ids = _group_id_to_vec_ids[group_id];
            if (vec_ids.empty())
               continue;

            auto &coverage = _label_nav_graph->covered_sets[group_id];
            coverage.clear();
            coverage.reserve(vec_ids.size());
            coverage.insert(coverage.end(), vec_ids.begin(), vec_ids.end());
         }

         // Step 2: 找出所有叶子节点（出度为 0）
         std::vector<IdxType> q;
         q.reserve((size_t)_num_groups);
         std::vector<int> out_degree(_num_groups + 1, 0); // 复制一份出度用于拓扑传播

         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            out_degree[group_id] = _label_nav_graph->out_neighbors[group_id].size();
            if (out_degree[group_id] == 0)
               q.push_back(group_id);
         }
         std::cout << "- Number of leaf nodes: " << q.size() << std::endl;

         // Step 3: 自底向上合并集合
         for (size_t qi = 0; qi < q.size(); ++qi)
         {
            IdxType current = q[qi];
            const auto &current_set = _label_nav_graph->covered_sets[current];

            for (auto parent : _label_nav_graph->in_neighbors[current])
            {
               // 将当前节点的集合合并到父节点中
               auto &parent_set = _label_nav_graph->covered_sets[parent];
               if (!current_set.empty())
               {
                  parent_set.insert(parent_set.end(), current_set.begin(), current_set.end());
               }

               // 减少父节点剩余未处理的子节点数
               out_degree[parent]--;
               if (out_degree[parent] == 0)
               {
                  q.push_back(parent);
               }
            }
         }

         // legacy path needs dedup because DAG may merge through multiple paths.
#pragma omp parallel for schedule(dynamic, 64) num_threads(coverage_threads)
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            auto &coverage = _label_nav_graph->covered_sets[group_id];
            if (coverage.size() <= 1)
               continue;
            std::sort(coverage.begin(), coverage.end());
            coverage.erase(std::unique(coverage.begin(), coverage.end()), coverage.end());
         }
      }

      // Step 4: 计算最终覆盖率
      _label_nav_graph->coverage_ratio.resize(_num_groups + 1, 0.0);
#pragma omp parallel for schedule(static) num_threads(coverage_threads)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         size_t covered_count = _label_nav_graph->covered_sets[group_id].size();
         _label_nav_graph->coverage_ratio[group_id] = static_cast<double>(covered_count) / _num_points;
      }

      // Step 5: 输出时间
      _cal_coverage_ratio_time = std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() - start_time)
                                     .count();
      std::cout << "- Finish in " << _cal_coverage_ratio_time << " ms" << std::endl;
   }
   // =====================================end LNG中每个f覆盖率计算=========================================

   // =====================================begin 计算LNG中后代的个数=========================================
   // fxy_add：使用广度优先搜索（BFS）来确保所有后代都被找到
   void UniNavGraph::get_descendants_info()
   {
      std::cout << "Calculating descendants info (using corrected BFS method)..." << std::endl;
      using PairType = std::pair<IdxType, int>;

      std::vector<PairType> descendants_num(_num_groups + 1);
      std::vector<std::vector<IdxType>> descendants_set(_num_groups + 1);

      auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel
      {
#pragma omp for schedule(dynamic, 16)
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            // 线程本地复用：避免每个 group 反复分配 visited/queue/set。
            thread_local std::vector<uint32_t> visited_epoch;
            thread_local uint32_t epoch = 1u;
            thread_local std::vector<IdxType> q;
            thread_local std::vector<IdxType> discovered;

            if (visited_epoch.size() != _num_groups + 1)
            {
               visited_epoch.assign(_num_groups + 1, 0u);
               epoch = 1u;
               q.clear();
               discovered.clear();
               q.reserve(256);
               discovered.reserve(256);
            }
            if (epoch == 0u)
            {
               std::fill(visited_epoch.begin(), visited_epoch.end(), 0u);
               epoch = 1u;
            }

            q.clear();
            discovered.clear();

            visited_epoch[group_id] = epoch;
            q.push_back(group_id);

            for (size_t head = 0; head < q.size(); ++head)
            {
               const IdxType u = q[head];
               for (const auto &v : _label_nav_graph->out_neighbors[u])
               {
                  if (visited_epoch[v] == epoch)
                     continue;
                  visited_epoch[v] = epoch;
                  discovered.push_back(v);
                  q.push_back(v);
               }
            }

            descendants_num[group_id] = PairType(group_id, (int)discovered.size());
            auto &dst = descendants_set[group_id];
            dst = discovered;

            ++epoch;
         }
      }

      // 单线程写入类成员变量
      _label_nav_graph->_lng_descendants_num = std::move(descendants_num);
      _label_nav_graph->_lng_descendants = std::move(descendants_set);

      // 计算平均后代个数
      double total_descendants = 0;
      for (const auto &pair : _label_nav_graph->_lng_descendants_num)
      {
         total_descendants += pair.second;
      }
      _label_nav_graph->avg_descendants = _num_groups > 0 ? total_descendants / _num_groups : 0.0;
      _cal_descendants_time = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
      std::cout << "- Finish in " << _cal_descendants_time << " ms" << std::endl;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
      std::cout << "- Average number of descendants per group: " << _label_nav_graph->avg_descendants << std::endl;
   }

   // =====================================end 计算LNG中后代的个数=========================================

   // =====================================begin 添加新的跨组边=========================================
   void UniNavGraph::finalize_intra_group_graphs()
   {
      std::cout << "Finalizing intra-group graphs by converting neighbor IDs to global..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      add_offset_for_uni_nav_graph();
      auto duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << duration << " ms" << std::endl;
   }

   void UniNavGraph::add_new_distance_oriented_edges(
       const std::string &dataset,
       uint32_t num_threads,
       ANNS::AcornInUng new_cross_edge) // MODIFIED: 参数类型改为 const std::string&
   {
      // 防御性断言
      assert(_base_storage != nullptr && "Error: _base_storage is not initialized!");
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph is not initialized! Did you call build_label_nav_graph()?");
      assert(!_vamana_instances.empty() && "Error: _vamana_instances is empty! Did you call build_graph_for_all_groups()?");
      assert(_vamana_instances.size() == _num_groups + 1 && "Error: _vamana_instances has incorrect size!");

      // 根据传入的字符串策略决定执行哪些步骤
      bool add_acorn_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_acorn");
      bool add_guarantee_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_guarantee");
      auto all_start_time = std::chrono::high_resolution_clock::now();

      // 提前清空新边集合，无论执行哪个分支都需要
      _my_new_edges_set.clear();

      // ==========================================================================================
      // PART 1 - ACORN 边构建逻辑 (Steps 1-3)
      // ==========================================================================================
      if (add_acorn_edges)
      {
         std::cout << "\n========================================================" << std::endl;
         std::cout << "--- Augmenting Graph with New Distance-Oriented Edges (ACORN) ---" << std::endl;
         std::cout << "========================================================" << std::endl;
         auto acorn_part_start_time = std::chrono::high_resolution_clock::now();

         const size_t MAX_ID_SAMPLES = 15;
         const size_t SAMPLING_THRESHOLD = 100000;

         // FXY_ADD: Step 1 (识别、采样) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 1: Identifying & Sampling Query Vectors] ---" << std::endl;
         auto step1_start_time = std::chrono::high_resolution_clock::now();

         // --- Step 0: 构建反向ID映射 ---
         std::cout << "[Step 0] Building reverse ID map for safe lookup..." << std::endl;
         std::unordered_map<IdxType, IdxType> old_to_new_map;
         old_to_new_map.reserve(_num_points);
         for (IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            old_to_new_map[_new_to_old_vec_ids[new_id]] = new_id;
         }
         std::cout << "  - Reverse ID map built successfully." << std::endl;

         // --- Step 1.1: 识别“概念根属性” ---
         std::cout << "[Step 1.1] Discovering conceptual root attributes via global coverage..." << std::endl;
         std::unordered_map<LabelType, size_t> attribute_counts;
         for (IdxType i = 0; i < _num_points; ++i)
         {
            for (const auto &label : _base_storage->get_label_set(i))
            {
               attribute_counts[label]++;
            }
         }
         std::unordered_set<LabelType> conceptual_root_labels;
         for (const auto &pair : attribute_counts)
         {
            if (static_cast<float>(pair.second) / _num_points >= new_cross_edge.root_coverage_threshold)
            {
               conceptual_root_labels.insert(pair.first);
            }
         }
         if (conceptual_root_labels.empty())
         {
            std::cerr << "  - WARNING: No conceptual root attributes found. Aborting edge addition." << std::endl;
            return;
         }
         std::cout << "  - Discovered " << conceptual_root_labels.size() << " conceptual root attributes." << std::endl;

         std::cout << "  	- Conceptual Root Labels are: { ";
         for (const auto &label : conceptual_root_labels)
         {
            std::cout << label << " ";
         }
         std::cout << "}" << std::endl;

         // --- Step 1.2: 根据层级比例识别候选组 ---
         std::cout << "[Step 1.2] Precisely identifying candidate groups using layer depth ratio..." << std::endl;
         size_t max_observed_depth = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (_group_id_to_label_set[group_id].size() > max_observed_depth)
            {
               max_observed_depth = _group_id_to_label_set[group_id].size();
            }
         }
         int target_max_depth = std::max(1, static_cast<int>(max_observed_depth * new_cross_edge.layer_depth_retio));
         std::cout << "  - Max observed layer depth in dataset: " << max_observed_depth << std::endl;
         std::cout << "  - Using layer_depth_ratio=" << new_cross_edge.layer_depth_retio << ", target max depth is " << target_max_depth << std::endl;
         std::unordered_set<IdxType> candidate_groups;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &label_set = _group_id_to_label_set[group_id];
            if (label_set.empty() || label_set.size() > target_max_depth)
               continue;
            for (const auto &label : label_set)
            {
               if (conceptual_root_labels.count(label))
               {
                  candidate_groups.insert(group_id);
                  break;
               }
            }
         }
         std::unordered_set<IdxType> topmost_group_ids;
         for (IdxType group_id : candidate_groups)
         {
            bool is_topmost = true;
            for (IdxType parent_id : _label_nav_graph->in_neighbors[group_id])
            {
               if (candidate_groups.count(parent_id))
               {
                  is_topmost = false;
                  break;
               }
            }
            if (is_topmost)
            {
               topmost_group_ids.insert(group_id);
            }
         }
         std::cout << "  - Identified " << candidate_groups.size() << " total candidate groups within target depth." << std::endl;
         std::cout << "  - Identified " << topmost_group_ids.size() << " true topmost groups among them." << std::endl;

         // --- 步骤 1.3: 将向量分入两个桶 ---
         std::cout << "[Step 1.3] Separating vectors into 'topmost' and 'subsequent' buckets..." << std::endl;
         std::vector<IdxType> topmost_vec_ids, subsequent_vec_ids;
         std::vector<float *> topmost_vec_data, subsequent_vec_data;
         std::vector<std::vector<LabelType>> topmost_vec_labels, subsequent_vec_labels;
         for (IdxType group_id : candidate_groups)
         {
            const auto &range = _group_id_to_range[group_id];
            const auto &label_set = _group_id_to_label_set[group_id];
            if (topmost_group_ids.count(group_id))
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  topmost_vec_ids.push_back(u);
                  topmost_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  topmost_vec_labels.push_back(label_set);
               }
            }
            else
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  subsequent_vec_ids.push_back(u);
                  subsequent_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  subsequent_vec_labels.push_back(label_set);
               }
            }
         }
         std::cout << "  - Topmost-layer bucket size: " << topmost_vec_ids.size() << " vectors." << std::endl;
         std::cout << "  - Subsequent-layer bucket size: " << subsequent_vec_ids.size() << " vectors." << std::endl;

         // --- 步骤 1.4: 应用分层抽样逻辑 ---
         std::cout << "[Step 1.4] Applying stratified sampling logic..." << std::endl;
         std::vector<IdxType> final_query_vec_ids;
         std::vector<float *> final_query_vec_data;
         std::vector<std::vector<LabelType>> final_query_vec_labels;
         size_t total_candidates_vecs = topmost_vec_ids.size() + subsequent_vec_ids.size();
         const auto original_topmost_vec_ids = topmost_vec_ids;

         if (total_candidates_vecs < SAMPLING_THRESHOLD || new_cross_edge.query_vector_ratio >= 1.0f)
         {
            std::cout << "  - Decision: Using ALL candidates without sampling." << std::endl;
            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_ids.insert(final_query_vec_ids.end(), subsequent_vec_ids.begin(), subsequent_vec_ids.end());
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_data.insert(final_query_vec_data.end(), subsequent_vec_data.begin(), subsequent_vec_data.end());
            final_query_vec_labels = std::move(topmost_vec_labels);
            final_query_vec_labels.insert(final_query_vec_labels.end(), subsequent_vec_labels.begin(), subsequent_vec_labels.end());
         }
         else
         {
            std::cout << "  - Decision: Performing stratified sampling." << std::endl;
            size_t target_total_count = static_cast<size_t>(total_candidates_vecs * new_cross_edge.query_vector_ratio);
            target_total_count = std::max(target_total_count, topmost_vec_ids.size());

            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_labels = std::move(topmost_vec_labels);

            size_t num_to_sample = target_total_count > final_query_vec_ids.size() ? target_total_count - final_query_vec_ids.size() : 0;
            if (num_to_sample > 0 && !subsequent_vec_ids.empty())
            {
               num_to_sample = std::min(num_to_sample, subsequent_vec_ids.size());
               std::vector<size_t> indices(subsequent_vec_ids.size());
               std::iota(indices.begin(), indices.end(), 0);
               std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

               for (size_t i = 0; i < num_to_sample; ++i)
               {
                  size_t original_idx = indices[i];
                  final_query_vec_ids.push_back(subsequent_vec_ids[original_idx]);
                  final_query_vec_data.push_back(subsequent_vec_data[original_idx]);
                  final_query_vec_labels.push_back(subsequent_vec_labels[original_idx]);
               }
            }
         }
         std::cout << "  - Final number of query vectors for edge augmentation: " << final_query_vec_ids.size() << std::endl;

         if (final_query_vec_ids.empty())
         {
            std::cout << "  - No vectors to process. Skipping edge addition." << std::endl;
            return;
         }

         // FXY_ADD: Step 1 计时结束
         _cross_edge_step1_time_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::high_resolution_clock::now() - step1_start_time)
                                         .count();
         std::cout << "--- [Finished Sub-step 1 in " << _cross_edge_step1_time_ms << " ms] ---" << std::endl;

         // --- 步骤 1.5: 执行 ACORN ---
         // FXY_ADD: Step 2 (执行ACORN) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 2: Executing ACORN Script] ---" << std::endl;
         auto step2_start_time = std::chrono::high_resolution_clock::now();
         std::cout << "[Step 1.5] Preparing input files and executing ACORN script..." << std::endl;
         std::string query_vec_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_vectors.fvecs";
         std::string query_attr_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_attrs.txt";
         write_fvecs(query_vec_path, final_query_vec_data, _base_storage->get_dim());
         write_labels_txt(query_attr_path, final_query_vec_labels);
         std::string output_neighbors_file = new_cross_edge.acorn_in_ung_output_path + "/R_neighbors_with_vectors.txt";
         std::stringstream cmd_ss;
         cmd_ss << "/home/fengxiaoyao/1023FilterVector/FilterVectorCode/ACORN/run_acorn_for_ung.sh "
                << dataset << " " << new_cross_edge.R_in_add_new_edge << " " << _base_storage->get_num_points() << " "
                << new_cross_edge.M << " " << new_cross_edge.M_beta << " " << new_cross_edge.gamma << " " << new_cross_edge.efs << " "
                << num_threads << " "
                << query_vec_path << " " << query_attr_path << " " << new_cross_edge.acorn_in_ung_output_path << " 1";
         std::string cmd = cmd_ss.str();
         std::cout << "  - Executing command: " << cmd << std::endl;
         int ret = system(cmd.c_str());
         if (ret != 0)
         {
            std::cerr << "  - FATAL ERROR: ACORN script execution failed! Check logs in " << new_cross_edge.acorn_in_ung_output_path << std::endl;
            return;
         }
         std::cout << "  - ACORN script finished successfully." << std::endl;

         // FXY_ADD: Step 2 计时结束
         _cross_edge_step2_acorn_time_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::high_resolution_clock::now() - step2_start_time)
                                               .count();
         std::cout << "--- [Finished Sub-step 2 in " << _cross_edge_step2_acorn_time_ms << " ms] ---" << std::endl;

         // --- 步骤 1.6 & 2: 解析、过滤并添加距离驱动边 ---
         // FXY_ADD: Step 3 (添加距离驱动边) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 3: Adding Distance-Oriented Edges] ---" << std::endl;
         auto step3_start_time = std::chrono::high_resolution_clock::now();

         std::cout << "[Step 1.6] Parsing ACORN output and generating candidates..." << std::endl;
         std::vector<NewEdgeCandidate> candidates;
         std::ifstream neighbor_file(output_neighbors_file);
         if (!neighbor_file.is_open())
         {
            std::cerr << "  - FATAL ERROR: Cannot open ACORN output file: " << output_neighbors_file << std::endl;
            return;
         }
         std::string line;
         long long current_query_local_idx = -1;
         IdxType u_id_new = 0;
         size_t intra_group_edges_skipped = 0;
         while (std::getline(neighbor_file, line))
         {
            if (line.rfind("Query: ", 0) == 0)
            {
               current_query_local_idx = std::stoll(line.substr(7));
               if (current_query_local_idx >= 0 && current_query_local_idx < final_query_vec_ids.size())
               {
                  u_id_new = final_query_vec_ids[current_query_local_idx];
               }
            }
            else if (line.rfind("NeighborID: ", 0) == 0)
            {
               std::stringstream line_ss(line);
               std::string token;
               IdxType v_id_old;
               line_ss >> token >> v_id_old;
               if (v_id_old < 0)
                  continue;
               auto it = old_to_new_map.find(v_id_old);
               if (it == old_to_new_map.end())
                  continue;
               IdxType v_id_new = it->second;
               IdxType group_u = _new_vec_id_to_group_id[u_id_new];
               IdxType group_v = _new_vec_id_to_group_id[v_id_new];
               if (group_u != group_v)
               {
                  float dist = _distance_handler->compute(
                      _base_storage->get_vector(u_id_new),
                      _base_storage->get_vector(v_id_new),
                      _base_storage->get_dim());
                  candidates.push_back({u_id_new, v_id_new, dist});
               }
               else
               {
                  intra_group_edges_skipped++;
               }
            }
         }
         neighbor_file.close();
         std::cout << "  - Generated " << candidates.size() << " initial cross-group candidates from ACORN results." << std::endl;
         std::cout << "  - Skipped " << intra_group_edges_skipped << " potential edges because they were intra-group." << std::endl;

         // ============================================================================================
         // FXY_FIX: 这是策略一的最终修复版本
         // ============================================================================================
         std::cout << "[Step 2] Adding new edges with smart selection logic and in-degree control (M=" << new_cross_edge.M_in_add_new_edge << ")..." << std::endl;

         // 1. 按“查询点”对所有候选邻居进行分组
         std::unordered_map<IdxType, std::vector<NewEdgeCandidate>> query_to_candidates;
         for (const auto &cand : candidates)
         {
            query_to_candidates[cand.from].push_back(cand);
         }

         _num_distance_oriented_edges = 0;

         // FXY_FIX: 1. 重新引入全局的入度计数器
         std::vector<std::atomic<IdxType>> new_edge_in_degree(_num_points);
         for (size_t i = 0; i < _num_points; ++i)
         {
            new_edge_in_degree[i] = 0;
         }

         // 2. 遍历每个查询点，独立地为其选择最佳出边
         for (auto const &[query_id, cand_list] : query_to_candidates)
         {
            std::vector<NewEdgeCandidate> hierarchical_candidates;
            std::vector<NewEdgeCandidate> non_hierarchical_candidates;

            IdxType from_group = _new_vec_id_to_group_id[query_id];

            // 3. 将候选邻居分为“层级邻居”和“普通邻居”两个桶
            for (const auto &cand : cand_list)
            {
               IdxType to_group = _new_vec_id_to_group_id[cand.to];
               const auto &children = _label_nav_graph->out_neighbors[from_group];
               bool is_hierarchical = false;
               for (const auto &true_child : children)
               {
                  if (to_group == true_child)
                  {
                     is_hierarchical = true;
                     break;
                  }
               }
               if (is_hierarchical)
               {
                  hierarchical_candidates.push_back(cand);
               }
               else
               {
                  non_hierarchical_candidates.push_back(cand);
               }
            }

            // 4. 对两个桶内的候选者分别按距离排序
            std::sort(hierarchical_candidates.begin(), hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });
            std::sort(non_hierarchical_candidates.begin(), non_hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });

            // 5. 智能添加边：同时考虑出度、入度限制
            size_t edges_added_for_this_query = 0;
            const size_t max_out_degree = new_cross_edge.M_in_add_new_edge;
            const size_t max_in_degree = new_cross_edge.M_in_add_new_edge; // 通常入度和出度限制相同

            // 5.1. 优先添加层级邻居
            for (const auto &cand : hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               // FXY_FIX: 2. 在添加边时检查目标节点的入度
               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     // FXY_FIX: 3. 成功添加后，才增加目标节点的入度计数
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }
            }

            // 5.2. 如果出度名额还有剩余，用普通邻居来填充
            for (const auto &cand : non_hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               // FXY_FIX: 2. 同样检查目标节点的入度
               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     // FXY_FIX: 3. 成功添加后，才增加目标节点的入度计数
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }
            }
         }

         std::cout << "  - Added a total of " << _num_distance_oriented_edges << " new distance-oriented edges to the graph with smart selection and in-degree control." << std::endl;

         // 诊断分析部分
         std::set<std::pair<IdxType, IdxType>> acorn_connected_group_pairs;
         for (const auto &edge_key : _my_new_edges_set)
         {
            IdxType from_id = edge_key >> 32;
            IdxType to_id = edge_key & 0xFFFFFFFF;
            acorn_connected_group_pairs.insert({_new_vec_id_to_group_id[from_id], _new_vec_id_to_group_id[to_id]});
         }

         std::cout << "\n--- [Cross-Edge Sub-step 3.5: ACORN Edge Connectivity Analysis] ---" << std::endl;
         // ... (诊断分析的 cout 代码完全不变) ...
         size_t total_required_hierarchical_links = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            total_required_hierarchical_links += _label_nav_graph->out_neighbors[group_id].size();
         }
         std::cout << "  - Total parent->child links in hierarchy: " << total_required_hierarchical_links << std::endl;
         std::cout << "  - ACORN edges created " << acorn_connected_group_pairs.size() << " unique group-to-group connections." << std::endl;

         size_t satisfied_hierarchical_links = 0;
         for (const auto &group_pair : acorn_connected_group_pairs)
         {
            IdxType parent_group = group_pair.first;
            IdxType child_group = group_pair.second;
            const auto &children = _label_nav_graph->out_neighbors[parent_group];
            bool is_hierarchical = false;
            for (const auto &true_child : children)
            {
               if (child_group == true_child)
               {
                  is_hierarchical = true;
                  break;
               }
            }
            if (is_hierarchical)
            {
               satisfied_hierarchical_links++;
            }
         }
         std::cout << "  - Among them, " << satisfied_hierarchical_links << " connections are valid parent->child links." << std::endl;
         double satisfaction_rate = total_required_hierarchical_links > 0 ? (double)satisfied_hierarchical_links / total_required_hierarchical_links * 100.0 : 0.0;
         std::cout << "  - Satisfaction rate of hierarchical links by ACORN: " << satisfaction_rate << "%" << std::endl;
         std::cout << "--- [Finished ACORN Edge Analysis] ---\n"
                   << std::endl;

         // FXY_ADD: Step 3 计时结束
         _cross_edge_step3_add_dist_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                        std::chrono::high_resolution_clock::now() - step3_start_time)
                                                        .count();
         std::cout << "--- [Finished Sub-step 3 in " << _cross_edge_step3_add_dist_edges_time_ms << " ms] ---" << std::endl;

         // 计算并保存总的跨组边构建时间
         _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - acorn_part_start_time)
                                       .count();
      }

      // ==========================================================================================
      // PART 2 - 层级保障边修复逻辑 (Step 4)
      // ==========================================================================================
      if (add_guarantee_edges)
      {
         std::cout << "\n--- [Cross-Edge Sub-step 4: Adding Targeted Edges for Unconnected Children] ---" << std::endl;
         auto step4_start_time = std::chrono::high_resolution_clock::now();

         // 1. 初始化，注意SearchCacheList仅用于获取临时的搜索队列
         size_t max_group_size = 0;
         if (_num_groups > 0)
         {
            for (auto group_id = 1; group_id <= _num_groups; ++group_id)
               max_group_size = std::max(max_group_size, static_cast<size_t>(_group_id_to_range[group_id].second - _group_id_to_range[group_id].first));
         }
         SearchCacheList search_cache_list(1, max_group_size, _Lbuild);

         // 2. 准备容器存储补充边
         std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges_hierarchical(_num_groups + 1);

         // 3. 以单线程模式遍历所有组
         std::cout << "  - Checking for and repairing broken parent->child links (in single-thread mode)..." << std::endl;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &cur_range = _group_id_to_range[group_id];
            if (cur_range.first >= cur_range.second || _label_nav_graph->out_neighbors[group_id].empty())
            {
               continue;
            }

            std::unordered_set<IdxType> connected_groups;
            for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            {
               for (const auto &neighbor : _graph->neighbors[i])
                  connected_groups.insert(_new_vec_id_to_group_id[neighbor]);
            }

            for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            {
               if (connected_groups.find(out_group_id) == connected_groups.end())
               {
                  const auto &out_range = _group_id_to_range[out_group_id];
                  size_t child_group_size = out_range.second - out_range.first;
                  if (child_group_size == 0)
                     continue;
                  if (child_group_size <= _max_degree)
                  { // 小规模组逻辑（蛮力）
                     IdxType query_vec_id = _group_entry_points[group_id];
                     const char *query = _base_storage->get_vector(query_vec_id);
                     float min_dist = std::numeric_limits<float>::max();
                     IdxType best_neighbor_id = -1;
                     for (IdxType i = out_range.first; i < out_range.second; ++i)
                     {
                        float dist = _distance_handler->compute(query, _base_storage->get_vector(i), _base_storage->get_dim());
                        if (dist < min_dist)
                        {
                           min_dist = dist;
                           best_neighbor_id = i;
                        }
                     }
                     if (best_neighbor_id != -1)
                     {
                        additional_edges_hierarchical[group_id].emplace_back(query_vec_id, best_neighbor_id);
                     }
                  }
                  else
                  { // 大规模组逻辑：使用手动图搜索，并配合正确的“已访问”集合
                     IdxType cnt = 0;
                     for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
                     {
                        const char *query_vec = _base_storage->get_vector(vec_id);
                        auto search_cache = search_cache_list.get_free_cache();
                        search_cache->search_queue.clear();
                        std::unordered_set<IdxType> visited_nodes_global;
                        IdxType child_group_entry_point = _group_entry_points[out_group_id];
                        float dist = _distance_handler->compute(query_vec, _base_storage->get_vector(child_group_entry_point), _base_storage->get_dim());

                        visited_nodes_global.insert(child_group_entry_point);
                        search_cache->search_queue.insert(child_group_entry_point, dist);
                        while (search_cache->search_queue.has_unexpanded_node())
                        {
                           const Candidate &cur = search_cache->search_queue.get_closest_unexpanded();
                           for (auto neighbor_id : _graph->neighbors[cur.id])
                           {
                              if (neighbor_id >= out_range.first && neighbor_id < out_range.second)
                              {
                                 if (visited_nodes_global.find(neighbor_id) == visited_nodes_global.end())
                                 {
                                    visited_nodes_global.insert(neighbor_id);
                                    float d = _distance_handler->compute(query_vec, _base_storage->get_vector(neighbor_id), _base_storage->get_dim());
                                    search_cache->search_queue.insert(neighbor_id, d);
                                 }
                              }
                           }
                        }
                        for (size_t k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                        {
                           additional_edges_hierarchical[group_id].emplace_back(vec_id, search_cache->search_queue[k].id);
                           cnt++;
                        }
                        search_cache_list.release_cache(search_cache);
                     }
                  }

                  // // 不再区分大小组，总是执行以下逻辑
                  // IdxType query_vec_id = _group_entry_points[group_id];
                  // const char* query = _base_storage->get_vector(query_vec_id);
                  // float min_dist = std::numeric_limits<float>::max();
                  // IdxType best_neighbor_id = -1;

                  // for (IdxType i = out_range.first; i < out_range.second; ++i) {
                  //    float dist = _distance_handler->compute(query, _base_storage->get_vector(i), _base_storage->get_dim());
                  //    if (dist < min_dist) { min_dist = dist; best_neighbor_id = i; }
                  // }

                  // if (best_neighbor_id != -1) {
                  //    additional_edges_hierarchical[group_id].emplace_back(query_vec_id, best_neighbor_id);
                  // }
               }
            }
         }
         std::cout << std::endl;

         // 4. 单线程合并所有补充边...
         std::cout << "  - Merging newly created fill-in edges..." << std::endl;
         size_t num_added_hierarchical_edges = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            for (const auto &[from_id, to_id] : additional_edges_hierarchical[group_id])
            {
               uint64_t edge_key = (uint64_t)from_id << 32 | to_id;
               if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
               {
                  _graph->neighbors[from_id].push_back(to_id);
                  _my_new_edges_set.insert(edge_key);
                  num_added_hierarchical_edges++;
               }
            }
         }
         std::cout << "  - Merged " << num_added_hierarchical_edges << " new hierarchical edges to repair the graph." << std::endl;

         _cross_edge_step4_add_hierarchy_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                             std::chrono::high_resolution_clock::now() - step4_start_time)
                                                             .count();
         std::cout << "--- [Finished Sub-step 4 in " << _cross_edge_step4_add_hierarchy_edges_time_ms << " ms] ---" << std::endl;
      }

      auto total_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - all_start_time).count();
      std::cout << "\n--- Finished augmenting graph in " << total_time_ms << " ms. ---" << std::endl;
      std::cout << "========================================================" << std::endl;
   }

   // =====================================begin 添加新的跨组边=========================================

//    void UniNavGraph::build_label_nav_graph()
//    {
//       std::cout << "Building label navigation graph... " << std::endl;
//       auto start_time = std::chrono::high_resolution_clock::now();
//       _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
//       omp_set_num_threads(_num_threads);

// // obtain out-neighbors
// #pragma omp parallel for schedule(dynamic, 256)
//       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
//       {
//          // if (group_id % 100 == 0)
//          //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
//          std::vector<IdxType> min_super_set_ids;
//          get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
//          _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
//       }

//       // obtain in-neighbors
//       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
//          for (auto each : _label_nav_graph->out_neighbors[group_id])
//             _label_nav_graph->in_neighbors[each].emplace_back(group_id);

//       _build_LNG_time = std::chrono::duration<double, std::milli>(
//                             std::chrono::high_resolution_clock::now() - start_time)
//                             .count();
//       std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
//    }

   // fxy_add : 打印信息的build_label_nav_graph
   void UniNavGraph::build_label_nav_graph()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

      // 监控变量
      std::atomic<size_t> processed_count{0};
      size_t total_groups = _num_groups;
      size_t log_interval = 100000; 

      // ===========================================================
      // Phase 1: Obtain Out-Neighbors (并行)
      // ===========================================================
      std::cout << "--- Phase 1: Calculating Out-Neighbors (Parallel) ---" << std::endl;
#pragma omp parallel for schedule(dynamic, 256)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 线程本地复用，减少每个group反复构造/析构临时vector的开销。
         thread_local std::vector<IdxType> min_super_set_ids;
         // 核心计算
         get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id].swap(min_super_set_ids);
         min_super_set_ids.clear();

         // 进度监控
         size_t current = ++processed_count;
         
         // 逻辑：
         // 1. 正常阶段：每 10 万打印一次
         // 2. 冲刺阶段（最后 1% 或 最后 10万）：每 1000 个打印一次，看到它在动
         bool normal_log = (current % log_interval == 0);
         bool final_sprint = (current > total_groups - 100000) && (current % 1000 == 0); 

         if (normal_log || final_sprint || current == total_groups)
         {
            #pragma omp critical 
            {
               double percentage = (double)current / total_groups * 100.0;
               std::cout << "[Phase 1] Processed " << current << " / " << total_groups 
                         << " (" << std::fixed << std::setprecision(4) << percentage << "%)" // 精度提高到4位
                         << std::endl;
            }
         }
      }

      // ===========================================================
      // Phase 2: Obtain In-Neighbors (串行)
      // ===========================================================
      std::cout << "--- Phase 2: Calculating In-Neighbors (Serial) ---" << std::endl;
      
      processed_count = 0; 
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (auto each : _label_nav_graph->out_neighbors[group_id])
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);

         size_t current = ++processed_count;
         if (current % log_interval == 0 || current == total_groups)
         {
             double percentage = (double)current / total_groups * 100.0;
             std::cout << "[Phase 2] Processed " << current << " / " << total_groups 
                       << " (" << std::fixed << std::setprecision(1) << percentage << "%)" 
                       << std::endl;
         }
      }

      _build_LNG_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "- Finished building LNG in " << _build_LNG_time << " ms" << std::endl;
   }


   // fxy_add:初始化求flag的几个数据结构
   void UniNavGraph::initialize_lng_descendants_coverage_bitsets()
   {
      std::cout << "Initializing LNG descendants and coverage bitsets..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _lng_descendants_bits.resize(_num_groups + 1);
      _covered_sets_bits.resize(_num_groups + 1);

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 初始化大小
         _lng_descendants_bits[group_id].resize(_num_groups);
         _covered_sets_bits[group_id].resize(_num_points);

         // 填充后代的group的集合
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         for (auto id : descendants)
         {
            _lng_descendants_bits[group_id].set(id);
         }

         // 填充覆盖的向量的集合
         const auto &coverage = _label_nav_graph->covered_sets[group_id];
         for (auto id : coverage)
         {
            _covered_sets_bits[group_id].set(id);
         }
      }
   }

   // fxy_add:初始化求flag的几个数据结构
   void UniNavGraph::initialize_roaring_bitsets()
   {
      std::cout << "enter initialize_roaring_bitsets" << std::endl;

      // 检查指针是否为空
      if (!_label_nav_graph)
      {
         std::cerr << "Error: _label_nav_graph is a null pointer!" << std::endl;
         return; // 或者抛出异常
      }

      _lng_descendants_rb.resize(_num_groups + 1);
      _covered_sets_rb.resize(_num_groups + 1);
#pragma omp parallel for schedule(dynamic, 128)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         const auto &coverage = _label_nav_graph->covered_sets[group_id];

         auto &desc_rb = _lng_descendants_rb[group_id];
         auto &cov_rb = _covered_sets_rb[group_id];
         if (!descendants.empty())
         {
            desc_rb.addMany(descendants.size(), descendants.data());
         }
         if (!coverage.empty())
         {
            cov_rb.addMany(coverage.size(), coverage.data());
         }
      }

      std::cout << "Roaring bitsets initialized." << std::endl;
   }

   // 将分组内的局部索引转换为全局索引
   void UniNavGraph::add_offset_for_uni_nav_graph()
   {
      omp_set_num_threads(_num_threads);
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto i = 0; i < _num_points; ++i)
         for (auto &neighbor : _graph->neighbors[i])
            neighbor += _group_id_to_range[_new_vec_id_to_group_id[i]].first;
   }

   UniNavGraph::CrossEdgeBackend UniNavGraph::resolve_cross_edge_backend() const
   {
      const int v = read_env_int_clamped("UNG_CROSS_EDGE_BACKEND", 1, 0, 1);
      return v == 0 ? CrossEdgeBackend::CPU : CrossEdgeBackend::GPU;
   }

   void UniNavGraph::build_cross_edges_generate_cpu_baseline(std::vector<SearchQueue> &cross_group_neighbors,
                                                             SearchCacheList &search_cache_list)
   {
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].empty())
            continue;

         const IdxType offset = _group_id_to_range[group_id].first;
         auto index = _vamana_instances[group_id];
         if (_num_cross_edges > _Lbuild)
         {
            std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
            exit(-1);
         }

         for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
         {
            const auto &range = _group_id_to_range[in_group_id];

#pragma omp parallel for schedule(dynamic, 1)
            for (IdxType vec_id = range.first; vec_id < range.second; ++vec_id)
            {
               const char *query = _base_storage->get_vector(vec_id);
               auto search_cache = search_cache_list.get_free_cache();
               index->iterate_to_fixed_point(query, search_cache);

               for (int k = 0; k < search_cache->search_queue.size(); ++k)
                  cross_group_neighbors[vec_id].insert(search_cache->search_queue[k].id + offset,
                                                       search_cache->search_queue[k].distance);
               search_cache_list.release_cache(search_cache);
            }
         }
      }
   }

   bool UniNavGraph::build_cross_edges_generate_gpu_optimized(std::vector<SearchQueue> &cross_group_neighbors,
                                                              double *h2d_ms_sum,
                                                              double *kernel_ms_sum,
                                                              double *d2h_ms_sum)
   {
      std::vector<IdxType> groups_to_process;
      groups_to_process.reserve(_num_groups);
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         if (!_label_nav_graph->in_neighbors[group_id].empty())
            groups_to_process.push_back(group_id);

      std::cout << "[cross_edges] backend=GPU, target_groups=" << groups_to_process.size() << std::endl;
      if (groups_to_process.empty())
         return true;

      bool gpu_prepared = false;
      try
      {
         gpu_prepare_all_vectors_on_device(h2d_ms_sum, nullptr);
         gpu_prepared = true;
         gpu_cross_groups_search_all_batched(groups_to_process, _base_storage->get_dim(), _num_cross_edges,
                                             cross_group_neighbors, h2d_ms_sum, kernel_ms_sum, d2h_ms_sum);
      }
      catch (const std::exception &e)
      {
         std::cerr << "[cross_edges][GPU] failed: " << e.what() << std::endl;
         prof_logf("[PROF] cross_edges.gpu_exception what=%s", e.what());
         if (gpu_prepared)
            gpu_release_all_vectors_on_device();
         return false;
      }
      catch (...)
      {
         std::cerr << "[cross_edges][GPU] failed: unknown exception" << std::endl;
         prof_logf("[PROF] cross_edges.gpu_exception what=unknown");
         if (gpu_prepared)
            gpu_release_all_vectors_on_device();
         return false;
      }

      if (gpu_prepared)
         gpu_release_all_vectors_on_device();
      return true;
   }

   void UniNavGraph::build_cross_group_edges()
   {
      std::cout << "Building cross-group edges ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // allocate memory for storaging cross-group neighbors
      std::vector<SearchQueue> cross_group_neighbors;
      cross_group_neighbors.resize(_num_points);
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         cross_group_neighbors[point_id].reserve(_num_cross_edges);

      // allocate memory for search caches
      size_t max_group_size = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
      SearchCacheList search_cache_list(_num_threads, max_group_size, _Lbuild);
      omp_set_num_threads(_num_threads);

      //my add
      //分项统计容器
      double gen_ms = 0.0;            // 候选生成总耗时（你的主要关注）
      double add_ms = 0.0;            // 生成“附加跨组边”耗时
      double merge_cross_ms = 0.0;    // 合并 cross_group_neighbors 到 _graph 的耗时
      double merge_add_ms = 0.0;      // 合并附加边的耗时
      double add_offset_ms = 0.0;     // add_offset_for_uni_nav_graph 的耗时
      //GPU细分累计槽（由 .cu 内的 cudaEvent 回传）
      double h2d_ms_sum = 0.0, kernel_ms_sum = 0.0, d2h_ms_sum = 0.0;

   {
      ScopedTimerMs t("cross_edges.generate_ms", &gen_ms);
      if (_index_name != "Vamana" && _index_name != "FixedPoolGPU")
      {
         std::cerr << "Error: invalid index name " << _index_name << std::endl;
         exit(-1);
      }

      // UNG_CROSS_EDGE_BACKEND: 0=CPU baseline, 1=GPU optimized.
      const CrossEdgeBackend backend = resolve_cross_edge_backend();
      // UNG_CROSS_EDGE_GPU_STRICT: 1=GPU失败直接报错; 0=失败自动回退CPU.
      const bool gpu_strict = read_env_int_clamped("UNG_CROSS_EDGE_GPU_STRICT", 0, 0, 1) == 1;

      // 可切换主线：
      // - CPU baseline: build_cross_edges_generate_cpu_baseline
      // - GPU optimized: build_cross_edges_generate_gpu_optimized（失败可回退）
      if (backend == CrossEdgeBackend::GPU)
      {
         const bool gpu_ok = build_cross_edges_generate_gpu_optimized(cross_group_neighbors,
                                                                       &h2d_ms_sum,
                                                                       &kernel_ms_sum,
                                                                       &d2h_ms_sum);
         if (!gpu_ok)
         {
            if (gpu_strict)
               throw std::runtime_error("GPU cross-edge generation failed in strict mode.");
            std::cout << "[cross_edges] fallback to CPU baseline path." << std::endl;
            build_cross_edges_generate_cpu_baseline(cross_group_neighbors, search_cache_list);
         }
      }
      else
      {
         std::cout << "[cross_edges] backend=CPU" << std::endl;
         build_cross_edges_generate_cpu_baseline(cross_group_neighbors, search_cache_list);
      }

   }

      // add additional edges
      std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges(_num_groups + 1);
{
      ScopedTimerMs t("cross_edges.additional_edges_ms", &add_ms);
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &cur_range = _group_id_to_range[group_id];
         std::unordered_set<IdxType> connected_groups;

         // obtain connected groups
         for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            for (IdxType j = 0; j < cross_group_neighbors[i].size(); ++j)
               connected_groups.insert(_new_vec_id_to_group_id[cross_group_neighbors[i][j].id]);

         // add additional cross-group edges for unconnected groups
         for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            if (connected_groups.find(out_group_id) == connected_groups.end())
            {
               IdxType cnt = 0;
               for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
               {
                  auto search_cache = search_cache_list.get_free_cache();
                  _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                  for (auto k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                  {
                     additional_edges[group_id].emplace_back(vec_id,
                                                             search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                     cnt += 1;
                  }
                  search_cache_list.release_cache(search_cache);
               }
            }
      }
}
      // add offset for uni-nav graph
   {
      ScopedTimerMs t("cross_edges.add_offset_ms", &add_offset_ms);  // 【添加】
      add_offset_for_uni_nav_graph();
   }

// merge cross-group edges
   {
      ScopedTimerMs t("cross_edges.merge_cross_ms", &merge_cross_ms);
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
            _graph->neighbors[point_id].emplace_back(cross_group_neighbors[point_id][k].id);
   }

// merge additional cross-group edges
   {
      ScopedTimerMs t("cross_edges.merge_additional_ms", &merge_add_ms);  // 【添加】
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (const auto &[from_id, to_id] : additional_edges[group_id])
            _graph->neighbors[from_id].emplace_back(to_id);
      }
   }

         std::cout << "[GPU GEMM] H2D(ms)=" << h2d_ms_sum
          << "  Kernel(ms)=" << kernel_ms_sum
          << "  D2H(ms)=" << d2h_ms_sum
          << std::endl;
          
      _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - start_time)
                                    .count();
      std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;

      //my add
      // 统一写一条汇总到文件（包含关键参数，方便复现实验）
      prof_logf("[PROF] cross_edges.summary total_ms=%.3f gen_ms=%.3f add_ms=%.3f add_offset_ms=%.3f merge_cross_ms=%.3f merge_add_ms=%.3f "
      "num_points=%u num_groups=%u num_cross_edges=%u Lbuild=%u threads=%u",
      _build_cross_edges_time, gen_ms, add_ms, add_offset_ms, merge_cross_ms, merge_add_ms,
      (unsigned)_num_points, (unsigned)_num_groups, (unsigned)_num_cross_edges, (unsigned)_Lbuild, (unsigned)_num_threads);
      // GPU细分总览（H2D/Kernel/D2H 累计）
      prof_logf("[PROF] cross_edges.gpu_breakdown_sum h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f",
         h2d_ms_sum, kernel_ms_sum, d2h_ms_sum);

   }

   /*void UniNavGraph::search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                            uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                            IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                            std::vector<std::bitset<10000001>> &bitmap)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // preparation
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // run queries
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;

         // for overlap or nofilter scenario
         if (scenario == "overlap" || scenario == "nofilter")
         {
            num_cmps[id] = 0;
            search_cache->visited_set.clear();
            cur_result.reserve(K);

            // obtain entry group
            std::vector<IdxType> entry_group_ids;
            if (scenario == "overlap")
               get_min_super_sets(_query_storage->get_label_set(id), entry_group_ids, false, false);
            else
               get_min_super_sets({}, entry_group_ids, true, true);

            // for each entry group
            for (const auto &group_id : entry_group_ids)
            {
               std::vector<IdxType> entry_points;
               get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

               // graph search and dump to current result
               num_cmps[id] += iterate_to_fixed_point(query, search_cache, id, entry_points, true, false);
               for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
                  cur_result.insert(search_cache->search_queue[k].id, search_cache->search_queue[k].distance);
            }

            // for the other scenarios: containment, equality
         }
         else
         {

            // obtain entry points
            auto entry_points = get_entry_points(_query_storage->get_label_set(id), num_entry_points, search_cache->visited_set);
            if (entry_points.empty())
            {
               num_cmps[id] = 0;
               for (auto k = 0; k < K; ++k)
                  results[id * K + k].first = -1;
               continue;
            }

            // graph search
            num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);
            cur_result = search_cache->search_queue;
         }

         // write results
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
               results[id * K + k].first = -1;
         }

         // clean
         search_cache_list.release_cache(search_cache);
      }
   }*/

   //fxy_add
   void UniNavGraph::warmup_selectors(uint32_t num_threads)
   {
      // 预热 Idea1 的 Trie 方法选择器
      if (_trie_method_selector != nullptr)
      {
         std::cout << "- Warming up Trie Method Selector (Idea1)..." << std::endl;
         // 创建一个符合模型输入尺寸的虚拟特征向量 ，特征数量为7
         std::vector<float> dummy_features1(7, 0.0f);

#pragma omp parallel for num_threads(num_threads)
         for (int i = 0; i < num_threads; ++i)
         {
            // 每个线程都调用一次 predict，触发其内部的初始化
            _trie_method_selector->predict(dummy_features1);
         }
         std::cout << "- Trie Method Selector is warm." << std::endl;
      }

      // 预热 Idea2 的 UNG/ACORN 选择器
      if (_ung_acorn_selector != nullptr)
      {
         std::cout << "- Warming up UNG/ACORN Selector (Idea2)..." << std::endl;
         // 创建一个符合模型输入尺寸的虚拟特征向量 ，特征数量为6
         std::vector<float> dummy_features2(6, 0.0f);

#pragma omp parallel for num_threads(num_threads)
         for (int i = 0; i < num_threads; ++i)
         {
            _ung_acorn_selector->predict(dummy_features2);
         }
         std::cout << "- UNG/ACORN Selector is warm." << std::endl;
      }
   }

   // fxy_add
   std::vector<float> UniNavGraph::calculate_idea1_features(const QueryStats &stats) const
   {
      std::vector<float> features;
      const float epsilon = 1e-9f;
      const float query_size = static_cast<float>(stats.query_length);
      const float cand_size = static_cast<float>(stats.candidate_set_size);
      const float trie_nodes = static_cast<float>(stats.trie_total_nodes);
      const float trie_cardinality = static_cast<float>(stats.trie_label_cardinality);
      const float trie_branching = static_cast<float>(stats.trie_avg_branching_factor);

      // const float log_cand_size = std::log1p(cand_size);   // 'Log_CandSize'
      // const float log_query_size = std::log1p(query_size); // 'Log_QuerySize'
      // const float cand_coverage_ratio = cand_size / (trie_nodes + epsilon);            // 'Cand_Coverage_Ratio'
      // const float query_cardinality_ratio = query_size / (trie_cardinality + epsilon); // 'Query_Cardinality_Ratio'
      // const float query_path_density = query_size * trie_branching; // 'Query_Path_Density'
      // const float avg_nodes_per_label = trie_nodes / (trie_cardinality + epsilon);
      // const float cand_selectivity = avg_nodes_per_label / (cand_size + epsilon); // 'Cand_Selectivity'
      // const float query_cand_ratio = query_size / (cand_size + epsilon);          // 'Query_Cand_Ratio'
      // const float cand_x_query_interaction = cand_size * query_size;    // 'Cand_x_Query_Interaction'
      // const float branching_x_cand_size = trie_branching * cand_size;   // 'Branching_x_CandSize'
      // const float branching_x_query_size = trie_branching * query_size; // 'Branching_x_QuerySize'
      // const float cand_size_sq = cand_size * cand_size;    // 'CandSize_sq'
      // const float query_size_sq = query_size * query_size; // 'QuerySize_sq'
      // const float log_cand_x_log_query = log_cand_size * log_query_size; // 'LogCand_x_LogQuery'

      if (_dataset == "celeba")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // 7. Cand_Selectivity   
      }
      else if (_dataset == "bigann")
      {
         features.reserve(7);
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // 7. Cand_Selectivity
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
      }
      else if (_dataset == "Genome")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // 7. Cand_Selectivity
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size);                                                           // 5. CandSize
         

         // features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         // features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         // features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         // features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         // features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         // features.push_back(cand_size);                                                           // 5. CandSize
         // features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // 7. Cand_Selectivity
      }
      else if (_dataset == "words")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                                          // 8. QuerySize
         features.push_back(query_size * trie_branching);                                         // 9. Query_Path_Density
         features.push_back(query_size / (trie_cardinality + epsilon));                           // 7. Query_Cardinality_Ratio
         features.push_back(trie_branching * query_size);                                         // 3. Branching_x_QuerySize
         features.push_back(query_size * query_size);                                             // 4. QuerySize_sq
      }
      else if (_dataset == "MTG")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize    
      }
      else if (_dataset == "Reviews")
      {
         features.reserve(7);
         features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                     // 8. QuerySize
         features.push_back(query_size * trie_branching);                    // 9. Query_Path_Density 
         features.push_back(query_size / (trie_cardinality + epsilon));      // 7. Query_Cardinality_Ratio
         features.push_back(trie_branching * query_size);                     // 3. Branching_x_QuerySize
         features.push_back(query_size * query_size);                        // 4. QuerySize_sq
         features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction  
      }
      else if (_dataset == "Amazon")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                     // 8. QuerySize
         features.push_back(query_size * trie_branching);                    // 9. Query_Path_Density 
         features.push_back(query_size / (trie_cardinality + epsilon));      // 7. Query_Cardinality_Ratio
         features.push_back(trie_branching * query_size);                    // 3. Branching_x_QuerySize
         features.push_back(query_size * query_size);                        // 4. QuerySize_sq

         // features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction
         // features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
         // features.push_back(query_size);                                     // 8. QuerySize
         // features.push_back(query_size * trie_branching);                    // 9. Query_Path_Density 
         // features.push_back(query_size / (trie_cardinality + epsilon));      // 7. Query_Cardinality_Ratio
         // features.push_back(trie_branching * query_size);                    // 3. Branching_x_QuerySize
         // features.push_back(query_size * query_size);                        // 4. QuerySize_sq
      }
      else if (_dataset == "Music")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                                          // 8. QuerySize
         features.push_back(query_size * trie_branching);                                         // 9. Query_Path_Density
         features.push_back(query_size / (trie_cardinality + epsilon));                           // 7. Query_Cardinality_Ratio
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio  
      }
      else if (_dataset == "openpmc")
      {
         features.reserve(7);
         features.push_back(query_size / (cand_size + epsilon));                                  // 7. Query_Cand_Ratio
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery                                                       // 8. QuerySize   
      }
      else if (_dataset == "AllNews")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         
         // features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         // features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         // features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         // features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         // features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         // features.push_back(cand_size);                                                           // 5. CandSize
         // features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
      }
      else if (_dataset == "Russian")
      {
         features.reserve(7);
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(query_size / (cand_size + epsilon));                                  // 7. Query_Cand_Ratio
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
      }
      else if (_dataset == "VariousImg")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery 
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio 
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
          
      }
      else if (_dataset == "Tiktok")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery 
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
  
      }
      else if (_dataset == "cord_19")
      {
         features.reserve(7);
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
      }
      else if (_dataset == "Laion")
      {
         features.reserve(7);
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction 
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery   
      }
      else if (_dataset == "BookReviews")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction 
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity

         // features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction 
         // features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         // features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         // features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         // features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         // features.push_back(cand_size);                                                           // 5. CandSize
         // features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
      }
      else {
         std::cerr << "Warning: Dataset name not recognized for Idea1 feature calculation." << std::endl;
      }
      return features;
   }

   // fxy_add
   std::vector<float> UniNavGraph::calculate_idea2_features(const QueryStats &stats) const
   {
      std::vector<float> features;
      const float epsilon = 1e-9f;
      const float num_descendants = static_cast<float>(stats.num_lng_descendants);
      const float total_coverage = stats.entry_group_total_coverage;
      const float num_entries = static_cast<float>(stats.num_entry_points);

      // const float log_num_entries = std::log1p(num_entries);         // 'LogNumEntries'
      // const float log_num_descendants = std::log1p(num_descendants); // 'LogNumDescendants'
      // const float log_total_coverage = std::log1p(total_coverage);   // 'LogTotalCoverage'
      // const float coverage_squared = total_coverage * total_coverage;      // 'CoverageSquared'
      // const float descendants_squared = num_descendants * num_descendants; // 'DescendantsSquared'
      // const float desc_cov_interaction = num_descendants * total_coverage;  // 'DescCovInteraction'
      // const float entries_cov_interaction = num_entries * total_coverage;   // 'EntriesCovInteraction'
      // const float entries_desc_interaction = num_entries * num_descendants; // 'EntriesDescInteraction'
      // const float desc_per_entry = num_descendants / (num_entries + epsilon); // 'DescPerEntry'
      // const float cov_per_entry = total_coverage / (num_entries + epsilon);   // 'CovPerEntry'

      if (_dataset == "celeba")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);   // 2. CoverageSquared
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(total_coverage);                    // 6. TotalCoverage
         features.push_back(std::log1p(total_coverage));        // 3. LogTotalCoverage
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
      }
      else if (_dataset == "bigann")
      {
         features.reserve(6);
         features.push_back(std::log1p(total_coverage));        // 3. LogTotalCoverage
         features.push_back(total_coverage * total_coverage);   // 2. CoverageSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(total_coverage);                    // 6. TotalCoverage
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared   
      }
      else if (_dataset == "Genome")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(total_coverage);                   // TotalCoverage
         features.push_back(std::log1p(total_coverage));       // LogTotalCoverage
         features.push_back(std::log1p(num_descendants));      // LogNumDescendants
         features.push_back(num_descendants * total_coverage); // DescCovInteraction
         features.push_back(num_entries * num_descendants);    // EntriesDescInteraction
         
         // features.push_back(num_entries * num_descendants);    // EntriesDescInteraction
         // features.push_back(num_descendants * total_coverage); // DescCovInteraction
         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(std::log1p(total_coverage));       // LogTotalCoverage
         // features.push_back(std::log1p(num_descendants));      // LogNumDescendants
         // features.push_back(total_coverage);                   // TotalCoverage
      }
      else if (_dataset == "words")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage); // DescCovInteraction
         features.push_back(num_entries * num_descendants);    // EntriesDescInteraction
         features.push_back(num_entries * total_coverage);     // EntriesCovInteraction
         features.push_back(std::log1p(total_coverage));       // LogTotalCoverage
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(total_coverage / (num_entries + epsilon)); // CovPerEntry
      }
      else if (_dataset == "MTG")
      {
         features.reserve(6);
         features.push_back(std::log1p(num_entries));           // 5. LogNumEntries
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(num_entries);                       // 4. NumEntries
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(num_entries * total_coverage);      // EntriesCovInteraction
         features.push_back(total_coverage);                           // 3. TotalCoverage
      }
      else if (_dataset == "Reviews")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(total_coverage);                           // 3. TotalCoverage 
         features.push_back(total_coverage / (num_entries + epsilon)); // CovPerEntry
         features.push_back(num_descendants * total_coverage); // DescCovInteraction
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
  
         // features.push_back(num_entries);                       // 4. NumEntries
         // features.push_back(std::log1p(num_entries));           // 5. LogNumEntries
         // features.push_back(num_entries * total_coverage);  // 1. EntriesCovInteraction
         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         // features.push_back(total_coverage);                           // 3. TotalCoverage  
      }
      else if (_dataset == "Amazon")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(num_descendants);       // NumDescendants

         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         // features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         // features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         // features.push_back(total_coverage);                           // 3. TotalCoverage
         // features.push_back(num_descendants);       // NumDescendants
      }
      else if (_dataset == "Music")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(num_descendants);       // NumDescendants
         features.push_back(total_coverage);                           // 3. TotalCoverage

         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         // features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         // features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         // features.push_back(total_coverage);                           // 3. TotalCoverage
         // features.push_back(num_descendants);       // NumDescendants
      }
      else if (_dataset == "openpmc")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(total_coverage);                           // 3. TotalCoverage
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(num_entries);                       // 4. NumEntries
         features.push_back(num_entries * num_descendants); // 6. EntriesDescInteraction
      }
      else if (_dataset == "AllNews")
      {
         features.reserve(6);
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(num_descendants);       // NumDescendants
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction

         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         // features.push_back(total_coverage);               // TotalCoverage
         // features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         // features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         // features.push_back(num_descendants);       // NumDescendants
      }
      else if (_dataset == "Russian")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage);  // DescCovInteraction
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(num_entries * total_coverage);      // EntriesCovInteraction
         features.push_back(std::log1p(total_coverage));        // LogTotalCoverage
         features.push_back(total_coverage);                    // TotalCoverage
         features.push_back(num_entries);                       // NumEntries   
      }
      else if (_dataset == "VariousImg")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(num_descendants / (num_entries + epsilon));// DescPerEntry

      }
      else if (_dataset == "Tiktok")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(total_coverage);               // TotalCoverage
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         
         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         // features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         // features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         // features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         // features.push_back(total_coverage);               // TotalCoverage
      }
      else if (_dataset == "cord_19")
      {
         features.reserve(6);
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants);       // NumDescendants
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(total_coverage * total_coverage);  // CoverageSquared    
      }
      else if (_dataset == "Laion")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants);       // NumDescendants
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(total_coverage);               // TotalCoverage
  

         // features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         // features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         // features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared  
         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(total_coverage);               // TotalCoverage
         // features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage    
      }
      else if (_dataset == "BookReviews")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         features.push_back(total_coverage);               // TotalCoverage 
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
           
         // features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         // features.push_back(total_coverage * total_coverage);  // CoverageSquared
         // features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         // features.push_back(std::log1p(total_coverage));               // 2. LogTotalCoverage
         // features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         // features.push_back(total_coverage);               // TotalCoverage   
      }
      else{
         std::cerr << "Warning: Dataset name not recognized for Idea2 feature calculation." << std::endl;
      }

      return features;
   }

   // fxy_add
   /*void UniNavGraph::calculate_query_features_only(
       std::shared_ptr<IStorage> query_storage,
       uint32_t num_threads,
       const std::string &output_csv_path,
       bool is_new_trie_method,
       bool is_rec_more_start)
   {
      std::cout << "Starting unified feature calculation for Idea1 & Idea2..." << std::endl;
      auto num_queries = query_storage->get_num_points();
      std::vector<QueryStats> query_stats(num_queries);

      // =================================================================
      // 阶段一: 并行计算所有查询的特征，并存储在内存中
      // =================================================================
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto &stats = query_stats[id];
         const auto &query_labels = query_storage->get_label_set(id);

         // --- 计算 Idea1 特征 ---
         stats.query_length = query_labels.size(); // QuerySize
         if (!query_labels.empty())
         {
            stats.candidate_set_size = _trie_index.get_candidate_count_for_label(query_labels.back()); // CandSize
         }
         // 复制Trie静态特征
         const auto &trie_metrics = _trie_static_metrics;
         stats.trie_total_nodes = trie_metrics.total_nodes;
         stats.trie_label_cardinality = trie_metrics.label_cardinality;
         stats.trie_avg_path_length = trie_metrics.avg_path_length;
         stats.trie_avg_branching_factor = trie_metrics.avg_branching_factor;

         // --- 计算 Idea2 特征 ---
         // 1. 获取入口组 (这是计算后续特征的前提)
         std::vector<IdxType> entry_group_ids;
         static std::atomic<int> temp_counter{0}; // 临时的计数器以满足函数签名
         get_min_super_sets_debug(query_labels, entry_group_ids, false, true, temp_counter, is_new_trie_method, is_rec_more_start, stats,false);

         // 2. 计算 NumDescendants 和 TotalCoverage
         if (!entry_group_ids.empty())
         {
            stats.num_entry_points = entry_group_ids.size(); // NumEntries
            stats.num_lng_descendants = [&]
            {
               roaring::Roaring desc;
               for (auto gid : entry_group_ids)
                  if (gid > 0 && gid <= _num_groups)
                     desc |= _lng_descendants_rb[gid];
               return desc.cardinality();
            }(); // NumDescendants

            stats.entry_group_total_coverage = [&]
            {
               roaring::Roaring cov;
               for (auto gid : entry_group_ids)
                  if (gid > 0 && gid <= _num_groups)
                     cov |= _covered_sets_rb[gid];
               return static_cast<float>(cov.cardinality()) / _num_points;
            }(); // TotalCoverage
         }
         else
         {
            stats.num_entry_points = 0;
            stats.num_lng_descendants = 0;
            stats.entry_group_total_coverage = 0.0f;
         }
      }
      std::cout << "In-memory feature calculation finished for " << num_queries << " queries." << std::endl;

      // =================================================================
      // 阶段二: 将内存中的所有统计数据一次性写入CSV文件
      // =================================================================
      std::cout << "Now writing all features to CSV file: " << output_csv_path << std::endl;
      std::ofstream outfile(output_csv_path);
      if (!outfile.is_open())
      {
         std::cerr << "FATAL ERROR: Could not open file to save features: " << output_csv_path << std::endl;
         return; // 提前退出
      }

      // 写入CSV表头 (包含Idea1和Idea2的所有特征)
      outfile << "QueryID,QuerySize,CandSize,TrieTotalNodes,TrieLabelCardinality,TrieAvgBranchingFactor,"
              << "NumEntries,NumDescendants,TotalCoverage\n";

      // 遍历所有查询的统计结果并写入文件
      for (size_t i = 0; i < query_stats.size(); ++i)
      {
         const auto &stats = query_stats[i];

         outfile << i << ","
                 // Idea1 features
                 << stats.query_length << ","
                 << stats.candidate_set_size << ","
                 << stats.trie_total_nodes << ","
                 << stats.trie_label_cardinality << ","
                 << stats.trie_avg_branching_factor << ","
                 // Idea2 features
                 << stats.num_entry_points << ","
                 << stats.num_lng_descendants << ","
                 << stats.entry_group_total_coverage << "\n";
      }

      outfile.close();
      std::cout << "Successfully saved all features to " << output_csv_path << std::endl;
   }*/

#include <future>   
#include <chrono>   
#include <ctime>    
#include <iomanip>  
#include <atomic>   
#include <algorithm> 

// fxy_add
void UniNavGraph::calculate_query_features_only(
    std::shared_ptr<IStorage> query_storage,
    uint32_t num_threads,
    const std::string &output_csv_path,
    bool is_new_trie_method,
    bool is_rec_more_start)
{
    std::cout << "Starting unified feature calculation (Monitor: Label [1] Completion)..." << std::endl;
    auto num_queries = query_storage->get_num_points();
    std::vector<QueryStats> query_stats(num_queries);

    // 设置慢查询监控阈值
    const double SLOW_QUERY_THRESHOLD_MS = 500000.0; 

    // 设置进度打印间隔
    size_t log_interval = std::max((size_t)1, (size_t)(num_queries * 0.05)); 
    std::atomic<size_t> processed_count{0}; 

    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (auto id = 0; id < num_queries; ++id)
    {
        auto &stats = query_stats[id];
        const auto &query_labels = query_storage->get_label_set(id);

        // --- 计算 Idea1 特征 ---
        stats.query_length = query_labels.size(); 
        if (!query_labels.empty())
        {
            stats.candidate_set_size = _trie_index.get_candidate_count_for_label(query_labels.back());
        }
        const auto &trie_metrics = _trie_static_metrics;
        stats.trie_total_nodes = trie_metrics.total_nodes;
        stats.trie_label_cardinality = trie_metrics.label_cardinality;
        stats.trie_avg_path_length = trie_metrics.avg_path_length;
        stats.trie_avg_branching_factor = trie_metrics.avg_branching_factor;

        // --- 计算 Idea2 特征 (带实时监控) ---
        std::vector<IdxType> entry_group_ids;
        static std::atomic<int> temp_counter{0}; 

        // 1. 启动异步计算
        auto future_result = std::async(std::launch::async, [&]() {
            get_min_super_sets_debug(query_labels, entry_group_ids, false, true, 
                                     temp_counter, is_new_trie_method, is_rec_more_start, stats, false);
        });

        // 2. 监控超时打印
        std::future_status status = future_result.wait_for(std::chrono::milliseconds((long long)SLOW_QUERY_THRESHOLD_MS));
        if (status == std::future_status::timeout)
        {
            #pragma omp critical
            {
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::cout << "\n[SLOW QUERY DETECTED - STILL RUNNING] " 
                          << std::put_time(std::localtime(&now_c), "%H:%M:%S") 
                          << " | ID: " << id << std::endl;
                
                std::cout << "  - Labels (" << query_labels.size() << "): [ ";
                for (size_t i = 0; i < query_labels.size(); ++i) {
                    std::cout << query_labels[i] << (i < query_labels.size() - 1 ? ", " : "");
                    if (i >= 10) { std::cout << "... "; break; } 
                }
                std::cout << " ]" << std::endl;
            }
        }

        // 3. 等待任务彻底完成
        future_result.get(); 

        // ============================================================
        // 特定标签 (Label 1) 完成时的监控打印
        // ============================================================
        if (query_labels.size() == 1 && query_labels[0] == 1)
        {
            #pragma omp critical
            {
                 // 获取当前时间
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                
                std::cout << "[Monitor] \033[1;32mTarget Query Finished\033[0m: ID " << id 
                          << " with Label [1] at " << std::put_time(std::localtime(&now_c), "%H:%M:%S") 
                          << " (Entry Groups: " << entry_group_ids.size() << ")"
                          << std::endl;
            }
        }
        // ============================================================

        // --- 后续 Bitmap 计算 ---
        if (!entry_group_ids.empty())
        {
            stats.num_entry_points = entry_group_ids.size();
            stats.num_lng_descendants = [&]
            {
                roaring::Roaring desc;
                for (auto gid : entry_group_ids)
                    if (gid > 0 && gid <= _num_groups)
                        desc |= _lng_descendants_rb[gid];
                return desc.cardinality();
            }(); 

            stats.entry_group_total_coverage = [&]
            {
                roaring::Roaring cov;
                for (auto gid : entry_group_ids)
                    if (gid > 0 && gid <= _num_groups)
                        cov |= _covered_sets_rb[gid];
                return static_cast<float>(cov.cardinality()) / _num_points;
            }(); 
        }
        else
        {
            stats.num_entry_points = 0;
            stats.num_lng_descendants = 0;
            stats.entry_group_total_coverage = 0.0f;
        }

        // 进度打印
        size_t current_processed = processed_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (current_processed % log_interval == 0 || current_processed == num_queries)
        {
            #pragma omp critical
            {
                float progress = (float)current_processed / num_queries * 100.0f;
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::cout << "[Progress " << std::put_time(std::localtime(&now_c), "%H:%M:%S") << "] " 
                          << current_processed << " / " << num_queries 
                          << " (" << std::fixed << std::setprecision(1) << progress << "%)" 
                          << std::endl;
            }
        }
    }
    
    std::cout << "In-memory feature calculation finished for " << num_queries << " queries." << std::endl;

    // =================================================================
    // 阶段二: 写入 CSV
    // =================================================================
    std::cout << "Now writing all features to CSV file: " << output_csv_path << std::endl;
    std::ofstream outfile(output_csv_path);
    if (!outfile.is_open())
    {
        std::cerr << "FATAL ERROR: Could not open file to save features: " << output_csv_path << std::endl;
        return; 
    }

    outfile << "QueryID,QuerySize,CandSize,TrieTotalNodes,TrieLabelCardinality,TrieAvgBranchingFactor,"
            << "NumEntries,NumDescendants,TotalCoverage\n";

    for (size_t i = 0; i < query_stats.size(); ++i)
    {
        const auto &stats = query_stats[i];

        outfile << i << ","
                << stats.query_length << ","
                << stats.candidate_set_size << ","
                << stats.trie_total_nodes << ","
                << stats.trie_label_cardinality << ","
                << stats.trie_avg_branching_factor << ","
                << stats.num_entry_points << ","
                << stats.num_lng_descendants << ","
                << stats.entry_group_total_coverage << "\n";
    }

    outfile.close();
    std::cout << "Successfully saved all features to " << output_csv_path << std::endl;
}




   //fxy_add: 在idea2之前，看看入口组个数，决定是否强制执行acorn
   std::optional<bool> UniNavGraph::check_idea2_heuristic_override(const std::string& dataset_name, size_t num_entry_groups) const
   {
      size_t entry_max= 1000;
      if (dataset_name == "AllNews") entry_max = 1000; 
         
      if (num_entry_groups > entry_max)
         {
               return true; // 返回 true = 强制使用 ACORN
         }
      return std::nullopt; 
   }

   //fxy_add:决定是否跳过 Entry Group 计算和 idea 决策
   std::optional<bool> UniNavGraph::check_pre_trie_heuristic(const std::string& dataset_name, size_t query_length, size_t candidate_set_size) const
   {

      if (query_length <= 1)
      {
         return true; // 强制使用 ACORN
      }

      if (dataset_name == "Russian" && query_length <=2)
      {
          //std::cout<<"2"<<std::endl;
          return true; 
      }
      
      return std::nullopt; // 没有命中任何规则，返回空，让 AI 来决策
   }

   size_t UniNavGraph::get_candidate_count_for_label(LabelType label) const
   {
      return _trie_index.get_candidate_count_for_label(label);
   }

// fxy_add 
    void UniNavGraph::thread_function(std::queue<int>& Qid_595,std::shared_ptr<IStorage> &query_storage,
                                   std::shared_ptr<DistanceHandler> &distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   bool is_idea2_available,
                                   bool is_new_trie_method, bool is_rec_more_start,
                                   bool is_ung_more_entry,
                                   int lsearch_start, int lsearch_step,
                                   int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold,
                                   int force_use_alg, bool is_bfs_filter, IdxType num_queries, 
                                   const std::vector<IdxType> &true_query_group_ids){
         omp_set_num_threads(1);

         lock_m.lock();
         int id = Qid_595.front();
         Qid_595.pop();
         lock_m.unlock();

         auto &stats = query_stats[id];
         auto total_search_start_time = std::chrono::high_resolution_clock::now();

         SearchCacheList search_cache_list(1, _num_points, Lsearch);
         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;
         cur_result.reserve(K);
         const auto &query_labels = _query_storage->get_label_set(id);
      

         // ======================= STAGE 1: DECISION MAKING =======================

         // --- 1.1 Initialize decision variables ---
         bool final_use_nT_true = false;       // Final decision for Trie method
         // bool final_use_acorn = false;         // Final decision for search framework
         int final_algo_choice = 0;            // 0: UNG, 1: ACORN(BFS), 2: ACORN(NoBFS)
         bool entry_groups_calculated = false; // Flag to prevent recalculation
         std::vector<IdxType> entry_group_ids;

         // --- 1.2 Calculate common features, needed for selectors ---
         stats.query_length = query_labels.size();
         if (!query_labels.empty())
         {
            stats.candidate_set_size = get_candidate_count_for_label(query_labels.back());
         }
         else
         {
            stats.candidate_set_size = 0; 
         }

         // --- PRE-TRIE 启发式规则 ---
         std::optional<bool> pre_heuristic_decision = std::nullopt;
         bool is_method3_mode = (force_use_alg == 0 && is_idea2_available && is_new_trie_method);
         if (is_method3_mode)
         {
            pre_heuristic_decision = check_pre_trie_heuristic(_dataset, stats.query_length, stats.candidate_set_size);
         }
         if (pre_heuristic_decision.has_value())
         {
            //final_use_acorn = pre_heuristic_decision.value(); // (设为 true)
            final_algo_choice = pre_heuristic_decision.value() ? 1 : 0;// 如果启发式决定用 ACORN (true)，默认给 1 (ACORN-Gamma/BFS)，否则给 0 (UNG)
            final_use_nT_true = true; 
            stats.is_idea1_used = final_use_nT_true;
            stats.is_idea2_used = static_cast<float>(final_algo_choice);
         }
         else if (force_use_alg == 0)
         {
            // --- Idea1 Selector Logic ---
            if (_trie_method_selector != nullptr && force_use_alg == 0 && is_new_trie_method)
            {
               auto idea1_flag_start_time = std::chrono::high_resolution_clock::now();

               // 1. 填充决策所需的 stats 信息
               const auto &trie_metrics = _trie_static_metrics;
               stats.trie_total_nodes = trie_metrics.total_nodes;
               stats.trie_label_cardinality = trie_metrics.label_cardinality;
               stats.trie_avg_path_length = trie_metrics.avg_path_length;
               stats.trie_avg_branching_factor = trie_metrics.avg_branching_factor;

               // 2. 调用独立函数计算特征向量
               std::vector<float> idea1_features = calculate_idea1_features(stats);

               // 3. 如果成功生成了特征 (即数据集匹配)，则用模型预测
               if (!idea1_features.empty())
               {
                  auto idea1_selector_start_time = std::chrono::high_resolution_clock::now();
                  final_use_nT_true = _trie_method_selector->predict(idea1_features);
                  stats.idea1_selector_pred_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - idea1_selector_start_time).count();
               }
               else
               {
                  // 若当前数据集无对应特征工程，则退回默认行为
                  std::cout<<"[Warning] No Idea1 features for dataset " << _dataset << ". Reverting to default behavior." << std::endl;
                  final_use_nT_true = is_new_trie_method;
               }
               stats.idea1_flag_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - idea1_flag_start_time).count();
            }
            else
            {
               // 如果模型不存在或被强制，则遵循原来的逻辑
               final_use_nT_true = is_new_trie_method;
            }
            // --- Idea2 Selector Logic ---
            if (is_idea2_available) //method2, method3
            {
               static std::atomic<int> temp_counter{0};
               QueryStats temp_stats;
               auto get_entry_group_start_time = std::chrono::high_resolution_clock::now();
               bool use_nT_for_group_get;
               if (force_use_alg == 2)
                  use_nT_for_group_get = true; // 强制设置为 true
               else
                  use_nT_for_group_get = (force_use_alg == 0) ? final_use_nT_true : is_new_trie_method;
               get_min_super_sets_debug(query_labels, entry_group_ids, false, true, temp_counter, use_nT_for_group_get, is_rec_more_start, temp_stats,is_new_trie_method);
               entry_groups_calculated = true;
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - get_entry_group_start_time).count();


               auto idea2_flag_start_time = std::chrono::high_resolution_clock::now();


               // 计算 Idea2 所需的原始特征并存入 stats
               if (!entry_group_ids.empty())
               {
                  stats.num_entry_points = entry_group_ids.size();
                  stats.num_lng_descendants = [&]
                  {
                     roaring::Roaring desc;
                     for (auto gid : entry_group_ids)
                        if (gid > 0 && gid <= _num_groups)
                           desc |= _lng_descendants_rb[gid];
                     return desc.cardinality();
                  }();
                  stats.entry_group_total_coverage = [&]
                  {
                     roaring::Roaring cov;
                     for (auto gid : entry_group_ids)
                        if (gid > 0 && gid <= _num_groups)
                           cov |= _covered_sets_rb[gid];
                     return static_cast<float>(cov.cardinality()) / _num_points;
                  }();
               }
               // 检查启发式规则 (仅在method3)
               std::optional<bool> heuristic_decision = std::nullopt;
               // if (is_method3_mode) // (force_use_alg=0 && is_idea2_available && is_new_trie_method)
               // {
               //     heuristic_decision = check_idea2_heuristic_override(_dataset, stats.num_entry_points); //
               // }
               if (heuristic_decision.has_value())
               {
                  //  final_use_acorn = heuristic_decision.value(); 
                   stats.idea2_selector_pred_time_ms = 0.0; 
               }
               else
               {
                  // 调用独立的特征计算函数
                  std::vector<float> idea2_features = calculate_idea2_features(stats);

                  // 使用模型进行预测 (仅当特征成功生成且未被强制时)
                  if (!idea2_features.empty() && _ung_acorn_selector != nullptr && force_use_alg == 0)
                  {
                     auto idea2_selector_start_time = std::chrono::high_resolution_clock::now();

                     //final_use_acorn = _ung_acorn_selector->predict(idea2_features);
                     float pred_val = _ung_acorn_selector->predict(idea2_features);
                     final_algo_choice = static_cast<int>(std::round(pred_val));

                     stats.idea2_selector_pred_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - idea2_selector_start_time).count();
                  }
                  // 如果模型不存在、数据集不匹配或被强制，final_use_acorn 保持默认值 false，最终由 force_use_alg 的逻辑覆盖
               }
               stats.idea2_flag_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - idea2_flag_start_time).count();

               
            }
         }

         // This block now only sets the final decision if force_use_alg is active.
         // The actual algorithm choice is made here, while the selector calculation may have happened above for data collection purposes.
         if (force_use_alg > 0)
         {
            switch (force_use_alg)
            {
            case 1:
               final_use_nT_true = false;
               final_algo_choice = 0;
               break;
            case 2:
               final_use_nT_true = true;
               final_algo_choice = 0;
               break;
            case 3: // ACORN (分为 Standard 和 Imp)
            case 4: // ACORN-1
               // 这里实现了对 Imp 的强制支持：
               // 如果脚本传参 is_bfs_filter=true，则对应类别 1
               // 如果脚本传参 is_bfs_filter=false，则对应类别 2 (Imp)
               if (is_bfs_filter) {
                  final_algo_choice = 1; //ACORN-gamma
               } else {
                  final_algo_choice = 2; //Imp
               }
               break;
            }
         }
         stats.is_idea1_used = final_use_nT_true;
         // stats.is_idea2_used = final_use_acorn;
         stats.is_idea2_used = static_cast<float>(final_algo_choice); 

         // ======================= STAGE 2: EXECUTION STAGE =======================

         // if (!entry_groups_calculated && (force_use_alg != 3 && force_use_alg != 4) && !pre_heuristic_decision.has_value())
         if (!entry_groups_calculated && final_algo_choice == 0)
         {
            auto get_entry_group_start_time = std::chrono::high_resolution_clock::now();
            static std::atomic<int> counter{0};
            get_min_super_sets_debug(query_labels, entry_group_ids, false, true, counter, final_use_nT_true, is_rec_more_start, stats,false);
            stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - get_entry_group_start_time).count();
         }

         // Apply entry point expansion logic if needed for the UNG path
         if (is_ung_more_entry && final_algo_choice == 0)
         {
            IdxType true_group_id = 0;
            if (id < true_query_group_ids.size())
            {
               true_group_id = true_query_group_ids[id];
            }
            const size_t extra_k = entry_group_ids.size() / 5;
            entry_group_ids = select_entry_groups(entry_group_ids, SelectionMode::SizeAndDistance, extra_k, 1.0, true_group_id);
         }
         stats.num_entry_points = entry_group_ids.size();

         if (final_algo_choice > 0)
         {
            auto search_time_start_ms = std::chrono::high_resolution_clock::now();
            std::shared_ptr<faiss::IndexACORNFlat> selected_acorn_index = nullptr; // <--- 使用一个临时指针

            if (force_use_alg == 4)
            {
               selected_acorn_index = _acorn_1_index;
            }
            else
            { // force_use_alg == 3 或自动选择
               selected_acorn_index = _acorn_index;
            }

            // --- Execute ACORN Search ---
            if (!selected_acorn_index)
            {
               std::cerr << "ERROR: ACORN index not loaded for query " << id << ". Skipping." << std::endl;
               for (auto k = 0; k < K; ++k)
                  results[id * K + k].first = -1;
               return;
            }

            int current_efs;

            // 2. 判断当前 Lsearch 处于哪个区间
            if (Lsearch <= lsearch_threshold)
            {
               // --- 处于慢速增长区 ---
               // 计算方式和原来一样，只是使用慢速步长
               current_efs = efs_start + ((Lsearch - lsearch_start) / lsearch_step) * efs_step_slow;
            }
            else
            {
               // --- 处于快速增长区 ---
               // 为了保证 efs 增长的连续性，计算分为两部分：
               
               // Part A: 先计算出在阈值点时的 efs 值
               int num_steps_in_slow_zone = (lsearch_threshold - lsearch_start) / lsearch_step;
               int efs_at_threshold = efs_start + num_steps_in_slow_zone * efs_step_slow;
               
               // Part B: 在阈值 efs 的基础上，加上快速增长的部分
               int num_steps_in_fast_zone = (Lsearch - lsearch_threshold) / lsearch_step;
               current_efs = efs_at_threshold + num_steps_in_fast_zone * efs_step_fast;
            }

            // 确保 efs 不会小于起始值 (安全检查)
            current_efs = std::max(current_efs, efs_start);

            stats.acorn_efs_used = current_efs;
            selected_acorn_index->acorn.efSearch = current_efs;

            const float *query_vector_float = reinterpret_cast<const float *>(query);
            std::vector<faiss::idx_t> result_original_ids(K);
            std::vector<float> result_dists(K);

            // [在 search 调用前，确定本次是否使用 BFS]
            // 如果是自动模式 (force=0): 看模型预测结果 (1是BFS, 2是Imp/NoBFS)
            // 如果是强制模式 (force>0): 看命令行参数 is_bfs_filter
            bool current_use_bfs_filter = true;
            if (force_use_alg == 0) {
               current_use_bfs_filter = (final_algo_choice == 1);
            } else {
               current_use_bfs_filter = is_bfs_filter;
            }

            
            if (pre_heuristic_decision.has_value() || force_use_alg == 3 || force_use_alg == 4)
            {// 强制使用ACORN-gamma时，调用ACORN原始的、基于倒排索引的过滤方法。

               // 将UNG的查询标签 (uint16_t) 转换为ACORN search函数期望的格式 (vector<vector<int>>)
               std::vector<std::vector<int>> query_attrs_for_acorn(1);
               query_attrs_for_acorn[0].assign(query_labels.begin(), query_labels.end());

               auto core_search_start_time = std::chrono::high_resolution_clock::now();
               selected_acorn_index->search_old_bitmap(
                   1, query_vector_float, K, result_dists.data(), result_original_ids.data(), query_attrs_for_acorn, nullptr, nullptr, nullptr, current_use_bfs_filter);
               stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - core_search_start_time).count();
            }
            else
            {          
               roaring::Roaring current_roaring_bitmap = compute_bitmap_from_groups(entry_group_ids);
               auto bitmap_start_time = std::chrono::high_resolution_clock::now();
               std::vector<char> filter_map(_num_points, 0);
               for (uint32_t original_id : current_roaring_bitmap)
               {
                  if (original_id < _num_points)
                  {
                     IdxType new_id = _old_to_new_vec_ids[original_id];
                     filter_map[new_id] = 1;
                  }
               }
               stats.bitmap_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - bitmap_start_time).count();
               auto core_search_start_time = std::chrono::high_resolution_clock::now();
               selected_acorn_index->search(1, query_vector_float, K, result_dists.data(), result_original_ids.data(), filter_map.data(), nullptr, nullptr, nullptr, current_use_bfs_filter);
               stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - core_search_start_time).count();
            }
            cur_result.clear();
            for (size_t i = 0; i < K; ++i)
            {
               if (result_original_ids[i] != -1)
               {
                  //cur_result.insert(_old_to_new_vec_ids[result_original_ids[i]], result_dists[i]);
                  cur_result.insert(result_original_ids[i], result_dists[i]);
               }
            }
            num_cmps[id] = 0;
            stats.search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - search_time_start_ms).count();
         }
         else
         {
            // --- Execute UNG Search ---
            auto search_time_start_ms = std::chrono::high_resolution_clock::now();
            search_cache->visited_set.clear();
            std::vector<IdxType> entry_points;
            for (const auto &group_id : entry_group_ids)
            {
               get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);
            }

            if (entry_points.empty())
            {
               stats.num_distance_calcs = 0;
               return;
            }

            auto core_search_start_time = std::chrono::high_resolution_clock::now();
            num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points, stats.num_nodes_visited);
            stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - core_search_start_time).count();

            stats.num_distance_calcs = num_cmps[id];
            cur_result = search_cache->search_queue;
            stats.search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - search_time_start_ms).count();
         }

         // ======================= STAGE 3: FINALIZE RESULTS =======================
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               // cur_result[k].id 无论来源是ACORN还是UNG，都统一是 "新ID".在这里一次性、正确地转换为 "原始ID"
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
            {
               results[id * K + k].first = -1;
            }
         }

         stats.time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - total_search_start_time).count();
         search_cache_list.release_cache(search_cache);
    }
   
   // fxy_add
   void UniNavGraph::search_hybrid(std::shared_ptr<IStorage> &query_storage,
                                   std::shared_ptr<DistanceHandler> &distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   bool is_idea2_available,
                                   bool is_new_trie_method, bool is_rec_more_start,
                                   bool is_ung_more_entry,
                                   int lsearch_start, int lsearch_step,
                                   int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold, 
                                   int force_use_alg,bool is_bfs_filter, const std::vector<IdxType> &true_query_group_ids)
   {
      // --- Initializations ---
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;
      query_stats.resize(num_queries);

      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }

      // const float COVERAGE_THRESHOLD = 0.8f;
      // const int MIN_LNG_DESCENDANTS_THRESHOLD = _num_points / 2.5;
      //SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

//       omp_set_num_threads(num_threads);
// #pragma omp parallel for schedule(dynamic, 1)
      std::queue<int> Qid_595;
      for (auto id = 0; id < num_queries; ++id)
         {Qid_595.push(id);}
            
      ThreadPool pool(num_threads);
      std::vector<std::future<int>> tp_results;
      for (auto id = 0; id < num_queries; ++id)
      {
         
         tp_results.emplace_back(
               
                pool.enqueue([this,&Qid_595,&query_storage,&distance_handler,&num_threads,&Lsearch,&num_entry_points,&scenario,&K, results,&num_cmps,&query_stats,&is_idea2_available,
                                   &is_new_trie_method, &is_rec_more_start,&is_ung_more_entry,&lsearch_start,&lsearch_step,
                                   &efs_start, &efs_step_slow,&efs_step_fast,&lsearch_threshold,
                                   &force_use_alg, &is_bfs_filter,&num_queries,  &true_query_group_ids] { // pass const type value j to thread; [] can be empty
                    this->thread_function(Qid_595,query_storage,distance_handler,num_threads,Lsearch,num_entry_points,scenario,K, results,num_cmps,query_stats,is_idea2_available,
                                   is_new_trie_method, is_rec_more_start,is_ung_more_entry,lsearch_start,lsearch_step,
                                   efs_start, efs_step_slow,efs_step_fast,lsearch_threshold,
                                   force_use_alg, is_bfs_filter, num_queries, true_query_group_ids);
                    return 1; // return to results; the return type must be the same with results
                }));   
      }
      for (auto &&tp_result : tp_results)			
         tp_result.get(); // result.get() makes sure this thread has been finished here;
   }

   std::vector<IdxType> UniNavGraph::get_entry_points(const std::vector<LabelType> &query_label_set,
                                                      IdxType num_entry_points, VisitedSet &visited_set)
   {
      std::vector<IdxType> entry_points;
      entry_points.reserve(num_entry_points);
      visited_set.clear();

      // obtain entry points for label-equality scenario
      if (_scenario == "equality")
      {
         auto node = _trie_index.find_exact_match(query_label_set);
         if (node == nullptr)
            return entry_points;
         get_entry_points_given_group_id(num_entry_points, visited_set, node->group_id, entry_points);

         // obtain entry points for label-containment scenario
      }
      else if (_scenario == "containment")
      {
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(query_label_set, min_super_set_ids);
         for (auto group_id : min_super_set_ids)
            get_entry_points_given_group_id(num_entry_points, visited_set, group_id, entry_points);
      }
      else
      {
         std::cerr << "Error: invalid scenario " << _scenario << std::endl;
         exit(-1);
      }

      return entry_points;
   }

   void UniNavGraph::get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                                     IdxType group_id, std::vector<IdxType> &entry_points)
   {
      const auto &group_range = _group_id_to_range[group_id];

      // not enough entry points, use all of them
      if (group_range.second - group_range.first <= num_entry_points)
      {
         for (auto i = 0; i < group_range.second - group_range.first; ++i)
            entry_points.emplace_back(i + group_range.first);
         return;
      }

      // add the entry point of the group
      const auto &group_entry_point = _group_entry_points[group_id];
      visited_set.set(group_entry_point);
      entry_points.emplace_back(group_entry_point);

      // randomly sample the other entry points
      for (auto i = 1; i < num_entry_points; ++i)
      {
         auto entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
         if (visited_set.check(entry_point) == false)
         {
            visited_set.set(entry_point);
            entry_points.emplace_back(i + group_range.first);
         }
      }
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                               IdxType target_id, const std::vector<IdxType> &entry_points,
                                               size_t &num_nodes_visited,
                                               bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
            neighbors = _graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            num_nodes_visited++; // search过程中遍历的点的个数

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   // fxy_add
   IdxType UniNavGraph::iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                                      IdxType target_id, const std::vector<IdxType> &entry_points,
                                                      bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_global_graph->neighbor_locks[cur.id]);
            neighbors = _global_graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   void UniNavGraph::save(std::string index_path_prefix, std::string results_path_prefix)
   {
      fs::create_directories(index_path_prefix);
      auto start_time = std::chrono::high_resolution_clock::now();

      // save meta data
      std::map<std::string, std::string> meta_data;
      statistics();
      meta_data["num_points"] = std::to_string(_num_points);
      meta_data["num_groups"] = std::to_string(_num_groups);
      meta_data["index_name"] = _index_name;
      meta_data["max_degree"] = std::to_string(_max_degree);
      meta_data["Lbuild"] = std::to_string(_Lbuild);
      meta_data["alpha"] = std::to_string(_alpha);
      meta_data["build_num_threads"] = std::to_string(_num_threads);
      meta_data["scenario"] = _scenario;
      meta_data["num_cross_edges"] = std::to_string(_num_cross_edges);
      meta_data["graph_num_edges"] = std::to_string(_graph_num_edges);
      meta_data["LNG_num_edges"] = std::to_string(_LNG_num_edges);
      meta_data["index_size(MB)"] = std::to_string(_index_size);
      meta_data["_index_size_add_rb(MB)"] = std::to_string(_index_size_add_rb);
      meta_data["index_time(ms)"] = std::to_string(_index_time - _build_roaring_bitsets_time);
      meta_data["index_time_add_rb(ms)"] = std::to_string(_index_time);
      meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
      meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
      meta_data["build_vector_attr_graph_time(ms)"] = std::to_string(_build_vector_attr_graph_time);
      meta_data["cal_descendants_time(ms)"] = std::to_string(_cal_descendants_time);
      meta_data["cal_coverage_ratio_time(ms)"] = std::to_string(_cal_coverage_ratio_time);
      meta_data["build_LNG_time(ms)"] = std::to_string(_build_LNG_time);
      meta_data["build_cross_edges_time(ms)"] = std::to_string(_build_cross_edges_time);
      // FXY_ADD: 保存详细的跨组边构建时间到 meta 文件
      meta_data["cross_edge_step1_time(ms)"] = std::to_string(_cross_edge_step1_time_ms);
      meta_data["cross_edge_step2_acorn_time(ms)"] = std::to_string(_cross_edge_step2_acorn_time_ms);
      meta_data["cross_edge_step3_add_dist_edges_time(ms)"] = std::to_string(_cross_edge_step3_add_dist_edges_time_ms);
      meta_data["cross_edge_step4_add_hierarchy_edges_time(ms)"] = std::to_string(_cross_edge_step4_add_hierarchy_edges_time_ms);

      // FXY_ADD: 计算并保存 Trie 静态指标
      std::cout << "Calculating and saving Trie static metrics..." << std::endl;
      TrieStaticMetrics trie_metrics = _trie_index.calculate_static_metrics();
      meta_data["trie_label_cardinality"] = std::to_string(trie_metrics.label_cardinality);
      meta_data["trie_total_nodes"] = std::to_string(trie_metrics.total_nodes);
      meta_data["trie_avg_path_length"] = std::to_string(trie_metrics.avg_path_length);
      meta_data["trie_avg_branching_factor"] = std::to_string(trie_metrics.avg_branching_factor);
      // Save the detailed label frequency distribution to its own file for easier analysis
      std::string trie_freq_filename = results_path_prefix + "trie_label_frequency.csv";
      std::ofstream freq_file(trie_freq_filename);
      freq_file << "LabelID,Frequency\n";
      for (const auto &pair : trie_metrics.label_frequency)
         freq_file << pair.first << "," << pair.second << "\n";
      freq_file.close();
      std::cout << "- Trie label frequency distribution saved to " << trie_freq_filename << std::endl;

      std::string meta_filename = index_path_prefix + "meta";
      write_kv_file(meta_filename, meta_data);

      // save build_time to csv
      std::string build_time_filename = results_path_prefix + "build_time.csv";
      std::ofstream build_time_file(build_time_filename);
      build_time_file << "Index Name,Build Time (ms)\n";
      build_time_file << "index_time" << "," << _index_time << "\n";
      build_time_file << "label_processing_time" << "," << _label_processing_time << "\n";
      build_time_file << "build_graph_time" << "," << _build_graph_time << "\n";
      build_time_file << "build_vector_attr_graph_time" << "," << _build_vector_attr_graph_time << "\n";
      build_time_file << "cal_descendants_time" << "," << _cal_descendants_time << "\n";
      build_time_file << "cal_coverage_ratio_time" << "," << _cal_coverage_ratio_time << "\n";
      build_time_file << "build_LNG_time" << "," << _build_LNG_time << "\n";
      build_time_file << "build_cross_edges_time" << "," << _build_cross_edges_time << "\n";
      // FXY_ADD: 保存详细的跨组边构建时间到 CSV 文件
      build_time_file << "cross_edge_step1_time" << "," << _cross_edge_step1_time_ms << "\n";
      build_time_file << "cross_edge_step2_acorn_time" << "," << _cross_edge_step2_acorn_time_ms << "\n";
      build_time_file << "cross_edge_step3_add_dist_edges_time" << "," << _cross_edge_step3_add_dist_edges_time_ms << "\n";
      build_time_file << "cross_edge_step4_add_hierarchy_edges_time" << "," << _cross_edge_step4_add_hierarchy_edges_time_ms << "\n";
      build_time_file.close();

      // save vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage->write_to_file(bin_file, label_file);

      // save group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      write_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // save group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      write_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // save group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      write_1d_vector(group_entry_points_filename, _group_entry_points);

      // save group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      write_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids saved." << std::endl;

      // save new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      write_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // save trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.save(trie_filename);

      // save graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph->save(graph_filename);

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph->save(global_graph_filename);

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      write_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // save LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      write_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);

      // save covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      write_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets saved." << std::endl;

      // save LNG descendant num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      write_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);

      // save LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      write_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // 保存 LNG 的核心图结构 (邻接表)
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      write_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors saved." << std::endl; // 增加一个打印，确认保存成功

      // save vector attr graph data
      std::string vector_attr_graph_filename = index_path_prefix + "vector_attr_graph";
      save_bipartite_graph(vector_attr_graph_filename);

      // // save _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // write_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // write_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);

      // save lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      save_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      save_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);

      // save acorn index
      std::cout << "\n--- Exporting reordered data for ACORN index building ---" << std::endl;

      // 1. 导出重排后的向量
      std::string reordered_vec_path = index_path_prefix + "reordered_vecs.fvecs";
      std::cout << "- Saving reordered vectors to: " << reordered_vec_path << std::endl;
      try
      {
         uint32_t dim = static_cast<uint32_t>(_base_storage->get_dim());
         std::ofstream out(reordered_vec_path, std::ios::binary);
         if (!out)
         {
            throw std::runtime_error("Cannot open file for writing: " + reordered_vec_path);
         }

         // 遍历所有已重排的向量并按 .fvecs 格式写入
         for (ANNS::IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            const float *vec_data = reinterpret_cast<const float *>(_base_storage->get_vector(new_id));
            // 先写入维度
            out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
            // 再写入向量数据
            out.write(reinterpret_cast<const char *>(vec_data), dim * sizeof(float));
         }
         out.close();
         std::cout << "- Reordered vectors saved successfully." << std::endl;
      }
      catch (const std::exception &e)
      {
         std::cerr << "Error while saving reordered vectors: " << e.what() << std::endl;
      }

      // 2. 导出与重排后向量顺序完全一致的标签
      std::string reordered_label_path = index_path_prefix + "reordered_labels.txt";
      std::ofstream reordered_label_file(reordered_label_path);
      if (!reordered_label_file.is_open())
      {
         std::cerr << "Error: Could not open file to save reordered labels: " << reordered_label_path << std::endl;
      }
      else
      {
         // 直接按新顺序 (0 to _num_points-1) 遍历已重排的 _base_storage
         for (ANNS::IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            const auto &label_set = _base_storage->get_label_set(new_id);
            for (size_t i = 0; i < label_set.size(); ++i)
            {
               reordered_label_file << label_set[i] << (i == label_set.size() - 1 ? "" : ",");
            }
            reordered_label_file << "\n";
         }
         reordered_label_file.close();
         std::cout << "- Reordered labels saved to: " << reordered_label_path << std::endl;
      }
      std::cout << "--- Finished exporting reordered data ---\n"
                << std::endl;
      // ========================== 修改部分结束 ==========================

      // print
      std::cout << "- Index saved in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   std::unordered_map<int, std::vector<int>> load_inverted_index(const std::string &input_path)
   {
      std::ifstream in(input_path, std::ios::binary);
      if (!in)
      {
         fprintf(stderr, "Error: Cannot open file for reading inverted index: %s\n", input_path.c_str());
         exit(1);
      }

      std::unordered_map<int, std::vector<int>> inverted_index;

      // 读取文件头：属性总数
      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));
      inverted_index.reserve(map_size);

      // 依次读取每个属性及其向量列表
      for (uint64_t i = 0; i < map_size; ++i)
      {
         int attr_id;
         uint64_t list_size;

         in.read(reinterpret_cast<char *>(&attr_id), sizeof(attr_id));
         in.read(reinterpret_cast<char *>(&list_size), sizeof(list_size));

         std::vector<int> vec_list(list_size);
         in.read(reinterpret_cast<char *>(vec_list.data()), list_size * sizeof(int));

         inverted_index[attr_id] = std::move(vec_list);
      }

      in.close();
      printf("Inverted index loaded successfully.\n");
      return inverted_index;
   }

   void UniNavGraph::load(std::string index_path_prefix, std::string selector_modle_prefix, const std::string &data_type, const std::string &acorn_index_path, const std::string &acorn_1_index_path, const std::string &dataset)
   {
      std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _dataset = dataset;

      // load meta data
      std::string meta_filename = index_path_prefix + "meta";
      auto meta_data = parse_kv_file(meta_filename);
      _num_points = std::stoi(meta_data["num_points"]);
      _num_groups = std::stoi(meta_data["num_groups"]);
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);

      // load vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(bin_file, label_file);

      // load group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // load group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      load_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // load group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      load_1d_vector(group_entry_points_filename, _group_entry_points);

      // load group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      load_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids loaded." << std::endl;

      // load new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      load_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // fxy_add: load old to new vec ids
      std::cout << "- Building reverse ID map (original_id -> new_id)..." << std::endl;
      _old_to_new_vec_ids.resize(_num_points);
      for (IdxType new_id = 0; new_id < _num_points; ++new_id)
      {
         IdxType old_id = _new_to_old_vec_ids[new_id];
         _old_to_new_vec_ids[old_id] = new_id;
      }
      std::cout << "- Reverse ID map built successfully." << std::endl;

      // load trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.load(trie_filename);
      // --- 预计算并缓存 Trie 静态指标 ---
      std::cout << "Pre-calculating and caching Trie static metrics..." << std::endl;
      _trie_static_metrics = _trie_index.calculate_static_metrics();
      std::cout << "- Caching complete." << std::endl;

      // load graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _graph->load(graph_filename);

      // fxy_add:load global graph data
      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _global_graph->load(global_graph_filename);

      // fxy_add: load global vamana entry point
      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      load_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // fxy_add: load LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      std::cout << "_label_nav_graph->coverage_ratio size: " << _label_nav_graph->coverage_ratio.size() << std::endl;
      load_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);
      std::cout << "LNG coverage ratio loaded." << std::endl;

      // fxy_add: load LNG descendants num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      load_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);
      std::cout << "LNG descendants num loaded." << std::endl;

      // fxy_add: load LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      load_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // fxy_add: load covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      load_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets loaded." << std::endl;

      // fxy_add: load LNG out_neighbors
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      load_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors loaded." << std::endl;

      // // fxy_add: load  _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // load_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::cout << "_lng_descendants_bits loaded." << std::endl;
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // load_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);
      // std::cout << "_covered_sets_bits loaded." << std::endl;

      // fxy_add: load lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      load_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::cout << "_lng_descendants_rb loaded." << std::endl;
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      load_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);
      std::cout << "_covered_sets_rb loaded." << std::endl;
      std::cout << " _label_nav_graph->out_neighbors.size() = " << _label_nav_graph->out_neighbors.size() << std::endl;
      std::cout << " _num_groups = " << _num_groups << std::endl;

      // load idea1 selector
      std::string model_path = selector_modle_prefix + "/idea1/idea1_selector_model_final.onnx";
      std::cout << "Loading Trie method selector model from " << model_path << " ..." << std::endl;
      try
      {
         if (fs::exists(model_path))
         {
            _trie_method_selector = std::make_unique<MethodSelector>(model_path);
            std::cout << "- Model loaded successfully." << std::endl;
         }
         else
         {
            std::cerr << "- WARNING: Model file not found. Will use default method passed by command line." << std::endl;
            _trie_method_selector = nullptr;
         }
      }
      catch (const std::exception &e)
      {
         std::cerr << "- ERROR: Failed to load model: " << e.what() << ". Will use default method." << std::endl;
         _trie_method_selector = nullptr;
      }

      // load idea2 selector
      std::string idea2_model_path = selector_modle_prefix + "/idea2/idea2_selector_model_final.onnx";
      std::cout << "Loading Idea2 Selector model from " << idea2_model_path << " ..." << std::endl;
      if (fs::exists(idea2_model_path))
      {
         std::cout << "- Loading Idea2 Selector model from: " << idea2_model_path << std::endl;
         try
         {
            _ung_acorn_selector = std::make_unique<MethodSelector>(idea2_model_path);
         }
         catch (const std::runtime_error &e)
         {
            std::cerr << "  - [ERROR] Failed to initialize Idea2 Selector: " << e.what() << std::endl;
            _ung_acorn_selector = nullptr;
         }
      }
      else
      {
         std::cout << "- [INFO] Idea2 Selector model not found at: " << idea2_model_path << ". Selector will be disabled." << std::endl;
         _ung_acorn_selector = nullptr;
      }

      // === load ACORN index ===
      // Declare a shared inverted index and a flag to track if it's loaded.
      std::unordered_map<int, std::vector<int>> shared_inverted_index;
      bool inverted_index_loaded = false;

      if (!acorn_index_path.empty() && fs::exists(acorn_index_path))
      {
         std::cout << "Loading REORDERED ACORN index from " << acorn_index_path << " ..." << std::endl;
         try
         {
            faiss::Index *raw_index = faiss::read_index(acorn_index_path.c_str());
            _acorn_index = std::shared_ptr<faiss::IndexACORNFlat>(dynamic_cast<faiss::IndexACORNFlat *>(raw_index));

            if (_acorn_index)
            {
               std::cout << "- Associating metadata directly (reordered)..." << std::endl;
               std::vector<std::vector<int>> metadata(_num_points);
               for (size_t new_id = 0; new_id < _num_points; ++new_id)
               {
                  const auto &label_set = _base_storage->get_label_set(new_id);
                  metadata[new_id].assign(label_set.begin(), label_set.end());
                  std::sort(metadata[new_id].begin(), metadata[new_id].end());
               }
               _acorn_index->set_metadata(metadata);
               std::cout << "- ACORN index loaded and metadata associated successfully." << std::endl;

               std::string inverted_index_path = acorn_index_path + ".inverted_index";
               if (fs::exists(inverted_index_path))
               {
                  std::cout << "- Loading inverted index from: " << inverted_index_path << std::endl;
                  shared_inverted_index = load_inverted_index(inverted_index_path);
                  _acorn_index->set_inverted_index(shared_inverted_index);
                  inverted_index_loaded = true;
                  std::cout << "- Inverted index loaded and set successfully for ACORN." << std::endl;
               }
               else
               {
                  std::cerr << "  - [WARNING] Inverted index file not found at: " << inverted_index_path
                            << ". ACORN's original filtering (force_use_alg=3) will not work." << std::endl;
               }
            }
            else
            {
               std::cerr << "ERROR: Failed to cast loaded index to faiss::IndexACORNFlat." << std::endl;
               delete raw_index;
            }
         }
         catch (const std::exception &e)
         {
            std::cerr << "ERROR: Exception caught while loading ACORN index: " << e.what() << std::endl;
            _acorn_index = nullptr;
         }
      }
      else if (!acorn_index_path.empty())
      {
         std::cerr << "Warning: ACORN index path provided, but file not found at: " << acorn_index_path << std::endl;
      }

      // --- load ACORN-1 index ---
      if (!acorn_1_index_path.empty() && fs::exists(acorn_1_index_path))
      {
         std::cout << "Loading REORDERED ACORN-1 index from " << acorn_1_index_path << " ..." << std::endl;
         try
         {
            faiss::Index *raw_index_1 = faiss::read_index(acorn_1_index_path.c_str());
            _acorn_1_index = std::shared_ptr<faiss::IndexACORNFlat>(dynamic_cast<faiss::IndexACORNFlat *>(raw_index_1));

            if (_acorn_1_index)
            {
               std::cout << "- Associating metadata directly with ACORN-1 index (reordered)..." << std::endl;
               std::vector<std::vector<int>> metadata(_num_points);
               for (size_t new_id = 0; new_id < _num_points; ++new_id)
               {
                  const auto &label_set = _base_storage->get_label_set(new_id);
                  metadata[new_id].assign(label_set.begin(), label_set.end());
                  std::sort(metadata[new_id].begin(), metadata[new_id].end());
               }
               _acorn_1_index->set_metadata(metadata);
               std::cout << "- ACORN-1 index loaded and metadata associated successfully." << std::endl;

               // Replace the old loading logic to reuse the inverted index.
               if (inverted_index_loaded)
               {
                  std::cout << "- Reusing the loaded inverted index for ACORN-1." << std::endl;
                  _acorn_1_index->set_inverted_index(shared_inverted_index);
                  std::cout << "- Inverted index set successfully for ACORN-1." << std::endl;
               }
               else
               {
                  std::cerr << "  - [WARNING] Inverted index could not be set for ACORN-1 because the primary inverted index file was not found." << std::endl;
               }
            }
            else
            {
               std::cerr << "ERROR: Failed to cast loaded index to faiss::IndexACORNFlat from " << acorn_1_index_path << std::endl;
               delete raw_index_1;
            }
         }
         catch (const std::exception &e)
         {
            std::cerr << "ERROR: Exception caught while loading ACORN-1 index: " << e.what() << std::endl;
            _acorn_1_index = nullptr;
         }
      }
      else if (!acorn_1_index_path.empty())
      {
         std::cerr << "Warning: ACORN-1 index file not found at: " << acorn_1_index_path << std::endl;
      }
      
      // print
      std::cout << "- Index loaded in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   /*void UniNavGraph::statistics()
   {
      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
            _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      _index_size_add_rb = _index_size; // 记录不含 Roaring Bitmap 部分的索引大小
      // fxy_add:计算为混合搜索策略预处理的 Roaring Bitmaps 的大小
      if (!_lng_descendants_rb.empty())
      {
         for (const auto &rb : _lng_descendants_rb)
         {
            _index_size_add_rb += rb.getSizeInBytes();
         }
      }
      if (!_covered_sets_rb.empty())
      {
         for (const auto &rb : _covered_sets_rb)
         {
            _index_size_add_rb += rb.getSizeInBytes();
         }
      }

      // return as MB
      _index_size /= 1024 * 1024;
      _index_size_add_rb /= 1024 * 1024;
   }*/
   void UniNavGraph::statistics()
   {
      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
               _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      // --- 统计 _label_nav_graph (LNG) 的大小 ---
      size_t lng_graph_size = 0;
      if (_label_nav_graph != nullptr)
      {
         // 1. 统计 out_neighbors (基于 .size())
         for (const auto &neighbors : _label_nav_graph->out_neighbors)
         {
               lng_graph_size += neighbors.size() * sizeof(ANNS::IdxType);
         }
         // 2. 统计 in_neighbors (基于 .size())
         for (const auto &neighbors : _label_nav_graph->in_neighbors)
         {
               lng_graph_size += neighbors.size() * sizeof(ANNS::IdxType);
         }
      }
      _index_size += lng_graph_size;

      // --- 统计 _group_id_to_vec_ids 的大小 ---
      size_t group_to_vec_size = 0;
      for (const auto &vec_ids : _group_id_to_vec_ids)
      {
         group_to_vec_size += vec_ids.size() * sizeof(ANNS::IdxType);
      }
      _index_size += group_to_vec_size;

      // fxy_add:计算为混合搜索策略预处理的 Roaring Bitmaps 的大小
      _index_size_add_rb = _index_size; 
      
      if (!_lng_descendants_rb.empty())
      {
         for (const auto &rb : _lng_descendants_rb)
         {
               _index_size_add_rb += rb.getSizeInBytes();
         }
      }
      if (!_covered_sets_rb.empty())
      {
         for (const auto &rb : _covered_sets_rb)
         {
               _index_size_add_rb += rb.getSizeInBytes();
         }
      }

      // return as MB
      _index_size /= 1024 * 1024;
      _index_size_add_rb /= 1024 * 1024;
   }
}
