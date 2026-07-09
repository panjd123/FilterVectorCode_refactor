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
   std::string acorn_index_path, acorn_1_index_path;
   std::string entry_group_provider_arg = "cpu_min_super_sets";
   std::string graph_search_backend_arg = "neighbor_list";
   ANNS::EntryGroupProviderImpl entry_group_provider = ANNS::EntryGroupProviderImpl::CpuMinSuperSets;
   ANNS::SearchGraphBackendImpl graph_search_backend = ANNS::SearchGraphBackendImpl::NeighborList;
   ANNS::IdxType K, num_entry_points;
   std::vector<ANNS::IdxType> Lsearch_list;
   uint32_t num_threads;
   bool is_new_method = false;                                 // true: use new method
   bool is_idea2_available = false;                            // true: use original ung
   bool is_new_trie_method = false, is_rec_more_start = false; // false:默认的UNG原始trie tree方法,true：递归；false:默认的root
   bool is_ung_more_entry = false;                             // false:默认的UNG原始entry point选择方法,true：更多entry points
   int num_repeats = 1;                                        // 默认重复1次
   int force_use_alg = 0;                                      // 0: auto, 1: UNG (nT=false), 2: UNG-nTtrue, 3: ACORN
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
      desc.add_options()("query_group_id_file", po::value<std::string>(&query_group_id_file)->required(),
                         "query_group_id_file");
      desc.add_options()("acorn_index_path", po::value<std::string>(&acorn_index_path)->default_value(""),
                         "acorn_index_path");
      desc.add_options()("acorn_1_index_path", po::value<std::string>(&acorn_1_index_path)->default_value(""),
                         "acorn_1_index_path");

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
      desc.add_options()("num_repeats", po::value<int>(&num_repeats)->default_value(1),
                         "Number of repeats for each Lsearch value");
      desc.add_options()("force_use_alg", po::value<int>(&force_use_alg)->required(),
                         "force_use_alg");
      desc.add_options()("lsearch_start", po::value<int>(&lsearch_start)->required(), "Lsearch start value");
      desc.add_options()("lsearch_step", po::value<int>(&lsearch_step)->required(), "Lsearch step value");
      desc.add_options()("efs_start", po::value<int>(&efs_start)->required(), "ACORN efs start value");
      desc.add_options()("efs_step_slow", po::value<int>(&efs_step_slow)->required(), "ACORN efs step value");
      desc.add_options()("efs_step_fast", po::value<int>(&efs_step_fast)->required(), "ACORN efs step value");
      desc.add_options()("lsearch_threshold", po::value<int>(&lsearch_threshold)->required(), "lsearch_threshold");
      desc.add_options()("entry_group_provider", po::value<std::string>(&entry_group_provider_arg)->default_value("cpu_min_super_sets"),
                         "Entry group provider: cpu_min_super_sets/cpu/0 or gpu_cover_frontier/gpu/1. "
                         "gpu_cover_frontier uses the production CUDA correct-cover provider when available.");
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
      if (selector_model_prefix.empty())
         selector_model_prefix = selector_model_prefix_legacy;
      if (selector_model_prefix.empty())
      {
         std::cerr << "Missing required option: selector_model_prefix (or deprecated selector_modle_prefix)" << std::endl;
         return -1;
      }
      entry_group_provider = ANNS::parse_entry_group_provider_impl(entry_group_provider_arg);
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

   // load index
   ANNS::UniNavGraph index(query_storage->get_num_points());
   index.load(index_path_prefix, selector_model_prefix, data_type, acorn_index_path, acorn_1_index_path,dataset);
   index.load_bipartite_graph(index_path_prefix + "vector_attr_graph");

   // 加载查询来源组ID文件
   std::vector<ANNS::IdxType> true_query_group_ids;
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

   // preparation
   auto num_queries = query_storage->get_num_points();
   std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
   auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
   ANNS::load_gt_file(gt_file, gt, num_queries, K);
   auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];

   std::vector<std::vector<ANNS::IdxType>> all_entry_group_ids =
       precompute_entry_group_ids(index, query_storage);
   const BitmapComparisonTiming bitmap_timing =
       run_bitmap_comparison(index, query_storage, all_entry_group_ids);
   (void)bitmap_timing;

   // calculate query features and save to CSV
   std::string features_csv_path = result_path_prefix + "query_features.csv";
   index.calculate_query_features_only(
       query_storage,
       num_threads,       
       features_csv_path, 
       true,              // is_new_trie_method
       true               // is_rec_more_start
   );

   // Warm-up selector
   std::cout << "\n--- Starting Warm-up Phase ---" << std::endl;
   index.warmup_selectors(num_threads);
   std::cout << "--- Warm-up Finished ---"<< std::endl;

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
                force_use_alg, entry_group_provider, graph_search_backend);

            index.search_hybrid(query_storage, distance_handler, runtime, results, num_cmps,
                                query_stats[repeat][LsearchId], true_query_group_ids);
         }
         auto time_cost = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();

         // 2. 计算每个独立查询的Recall
         for (int i = 0; i < num_queries; ++i)
            query_stats[repeat][LsearchId][i].recall = calculate_single_query_recall(gt + i * K, results + i * K, K);

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
      // 1. 修改表头，增加 Average_Efs 列
      summary_out << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n";
      for (auto const &[l_search, times] : time_per_lsearch)
      {
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

            // 3. 将 avg_efs 写入文件
            summary_out << l_search << "," << avg_efs << "," << avg_time << "," << avg_recall << "\n";
         }
      }
      summary_out.close();
      std::cout << "性能汇总 (平均efs/耗时/召回率) 已保存到: " << summary_file_path << std::endl;
   }
   else
   {
      std::cerr << "错误：无法打开文件 " << summary_file_path << " 进行写入" << std::endl;
   }

   // save query details
   std::ofstream detail_out(result_path_prefix + "query_details_repeat" + std::to_string(num_repeats) + ".csv");
   detail_out << "repeat,Lsearch,efs,QueryID,Time_ms,search_time_ms,core_search_time_ms,Recall,"         // 核心结果
              << "is_idea1_used,is_idea2_used,"                                                          // 使用的方法
              << "DistCalcs,NumNodeVisited,"                                                             // 性能指标
              << "MinSupersetT_ms,idea1SelT_ms,idea2SelT_ms,idea1_flag_ms,idea2_flag_ms,BitmapT_new_ms," // 耗时分解
              // --- Idea1 & Trie 特征 ---
              << "QuerySize,CandSize,"
              // --- Idea2 模型核心特征 ---
              << "NumEntries,"
              << "EntryGroupMatchedPoints,SpecialSearchEnabled,SpecialFreeUseRegular,"
              << "SpecialQueryMatchedPoints,SpecialQueryPoints,SpecialQueryTrivialPoints,SpecialQueryNontrivialPoints,"
              << "SpecialQueryRatio,SpecialQueryNontrivialRatio,SpecialQueryBlockCount,SpecialQueryTrivialBlockCount,SpecialQueryNontrivialBlockCount,"
	              << "SpecialEntryFreePoints,SpecialEntryRegularPoints,SpecialRegularExpanded,SpecialFreeExpanded,"
	              << "SpecialRegularInserted,SpecialFreeInserted,SpecialFreeUpgrades,"
	              << "SpecialRegularEdgesScanned,SpecialEdgesScanned,SpecialIntraEdgesScanned,SpecialInterEdgesScanned,"
	              << "SpecialHeavyEdgesEnabled,SpecialHeavyEdgesScanned,SpecialHeavyEdgesAccepted,"
	              << "SpecialRegularEdgesAccepted,SpecialEdgesAccepted,SpecialRegularDistCalcs,SpecialFreeDistCalcs"
              << "\n";
   for (int repeat = 0; repeat < num_repeats; repeat++)
   {
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         for (int i = 0; i < num_queries; ++i)
         {
            const auto &stats = query_stats[repeat][LsearchId][i];
            detail_out << repeat << ","
                       << Lsearch_list[LsearchId] << ","
                       << stats.acorn_efs_used << ","
                       << i << ","
                       << stats.time_ms << ","
                       << stats.search_time_ms << ","
                       << stats.core_search_time_ms << ","
                       << stats.recall << ","
                       << stats.is_idea1_used << ","
                       << stats.is_idea2_used << ","
                       << stats.num_distance_calcs << ","
                       << stats.num_nodes_visited << ","
                       << stats.get_min_super_sets_time_ms << ","
                       << stats.idea1_selector_pred_time_ms << ","
                       << stats.idea2_selector_pred_time_ms << ","
                       << stats.idea1_flag_time_ms << ","
                       << stats.idea2_flag_time_ms << ","
                       << stats.bitmap_time_ms << ","
                       // Idea1 & Trie 特征
                       << stats.query_length << ","
                       << stats.candidate_set_size << ","
                       // Idea2 模型核心特征
                       << stats.num_entry_points << ","
                       << stats.entry_group_matched_points << ","
                       << (stats.special_search_enabled ? 1 : 0) << ","
                       << (stats.special_free_use_regular ? 1 : 0) << ","
                       << stats.special_query_matched_points << ","
                       << stats.special_query_points << ","
                       << stats.special_query_trivial_points << ","
                       << stats.special_query_nontrivial_points << ","
                       << stats.special_query_ratio << ","
                       << stats.special_query_nontrivial_ratio << ","
                       << stats.special_query_block_count << ","
                       << stats.special_query_trivial_block_count << ","
                       << stats.special_query_nontrivial_block_count << ","
                       << stats.special_entry_free_points << ","
                       << stats.special_entry_regular_points << ","
                       << stats.special_regular_nodes_expanded << ","
                       << stats.special_free_nodes_expanded << ","
                       << stats.special_regular_candidates_inserted << ","
                       << stats.special_free_candidates_inserted << ","
                       << stats.special_free_upgrades << ","
                       << stats.special_regular_edges_scanned << ","
	                       << stats.special_edges_scanned << ","
	                       << stats.special_intra_edges_scanned << ","
	                       << stats.special_inter_edges_scanned << ","
	                       << (stats.special_heavy_edges_enabled ? 1 : 0) << ","
	                       << stats.special_heavy_edges_scanned << ","
	                       << stats.special_heavy_edges_accepted << ","
	                       << stats.special_regular_edges_accepted << ","
                       << stats.special_edges_accepted << ","
                       << stats.special_regular_distance_calcs << ","
                       << stats.special_free_distance_calcs << "\n";
         }
      }
   }

   detail_out.close();
   std::cout << "- all done" << std::endl;
   return 0;
}
