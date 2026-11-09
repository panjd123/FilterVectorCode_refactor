#include "common.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <queue>

#include <hnswlib/space_l2.h>

namespace {

struct RunRow {
    int repeat;
    int ef;
    double elapsed_time_ms;
    double recall;
    double qps;
};

struct QueryRow {
    std::size_t query_id;
    int ef;
    double search_time_ms;
    double recall;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 14) {
            std::cerr << "usage: " << argv[0]
                      << " <query.fvecs> <query_labels.txt> <schema.txt> <gt_ids.bin>"
                         " <index.bin> <result_dir> <topk> <num_repeats> <num_threads>"
                         " <search_mode> <hybrid_factor> <pruning_factor> <ef> [ef ...]\n";
            return 2;
        }
        const std::filesystem::path query_path = argv[1];
        const std::filesystem::path query_label_path = argv[2];
        const std::filesystem::path schema_path = argv[3];
        const std::filesystem::path groundtruth_path = argv[4];
        const std::filesystem::path index_path = argv[5];
        const std::filesystem::path result_dir = argv[6];
        const int topk = std::stoi(argv[7]);
        const int num_repeats = std::stoi(argv[8]);
        const int num_threads = std::stoi(argv[9]);
        const int search_mode = std::stoi(argv[10]);
        const float hybrid_factor = std::stof(argv[11]);
        const float pruning_factor = std::stof(argv[12]);
        if (topk <= 0 || num_repeats <= 0 || num_threads <= 0 ||
            search_mode < 0 || search_mode > 3) {
            throw std::runtime_error("invalid search arguments");
        }
        std::vector<int> ef_values;
        for (int argument = 13; argument < argc; ++argument) {
            const int ef = std::stoi(argv[argument]);
            if (ef <= 0) throw std::runtime_error("ef values must be positive");
            ef_values.push_back(ef);
        }
        if (ef_values.empty()) throw std::runtime_error("at least one ef value is required");

        rwalks_adapter::MappedFvecs queries(query_path);
        const auto schema = rwalks_adapter::load_schema(schema_path);
        const auto query_attributes = rwalks_adapter::load_dense_labels(
            query_label_path, queries.count(), schema, true);
        for (std::size_t query = 0; query < queries.count(); ++query) {
            const int* attributes = query_attributes.data() + query * schema.size();
            if (std::none_of(attributes, attributes + schema.size(), [](int value) { return value != 0; })) {
                throw std::runtime_error(
                    "RWalks does not support an empty filter at query " + std::to_string(query));
            }
        }
        const auto groundtruth = rwalks_adapter::load_groundtruth(
            groundtruth_path, queries.count(), topk);

        hnswlib::L2Space space(queries.dimension(), schema.size());
        auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &space, index_path.string(), false, 0, false);
        index->search_mode = search_mode;
        index->hybrid_factor_ = hybrid_factor;
        index->pron_factor_ = pruning_factor;
        std::cout << "loaded RWalks index: elements=" << index->cur_element_count.load()
                  << " dimension=" << queries.dimension()
                  << " attribute_dimension=" << schema.size()
                  << " search_mode=" << search_mode
                  << " hybrid_factor=" << hybrid_factor
                  << " pruning_factor=" << pruning_factor << '\n';
        rwalks_adapter::print_memory_stats(*index);

        std::filesystem::create_directories(result_dir);
        omp_set_num_threads(num_threads);
        std::vector<RunRow> run_rows;
        std::vector<QueryRow> query_rows;
        std::map<int, std::vector<RunRow>> rows_by_ef;
        std::vector<hnswlib::labeltype> result_ids(
            queries.count() * static_cast<std::size_t>(topk),
            static_cast<hnswlib::labeltype>(-1));

        for (int repeat = 0; repeat < num_repeats; ++repeat) {
            std::cout << "\n=== Repeat " << (repeat + 1) << '/' << num_repeats << " ===\n";
            for (int ef : ef_values) {
                index->ef_ = static_cast<std::size_t>(ef);
                std::fill(result_ids.begin(), result_ids.end(), static_cast<hnswlib::labeltype>(-1));
                const auto begin = std::chrono::steady_clock::now();
                double recall_sum = 0.0;
                std::vector<QueryRow> ef_query_rows(queries.count());
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1) reduction(+ : recall_sum)
                for (std::int64_t signed_query = 0;
                     signed_query < static_cast<std::int64_t>(queries.count()); ++signed_query) {
                    const std::size_t query = static_cast<std::size_t>(signed_query);
                    const auto query_begin = std::chrono::steady_clock::now();
                    auto results = index->searchKnn(
                        queries.row(query), static_cast<std::size_t>(topk), nullptr,
                        query_attributes.data() + query * schema.size(), false);
                    const auto query_end = std::chrono::steady_clock::now();
                    const double query_time_ms = std::chrono::duration<double, std::milli>(
                        query_end - query_begin).count();
                    std::size_t offset = query * static_cast<std::size_t>(topk) + results.size();
                    while (!results.empty()) {
                        --offset;
                        result_ids[offset] = results.top().second;
                        results.pop();
                    }
                    const double query_recall = rwalks_adapter::recall_at_k(
                        groundtruth.data() + query * topk,
                        result_ids.data() + query * topk, topk);
                    recall_sum += query_recall;
                    ef_query_rows[query] = {query, ef, query_time_ms, query_recall};
                }
                query_rows.insert(
                    query_rows.end(), ef_query_rows.begin(), ef_query_rows.end());
                const auto end = std::chrono::steady_clock::now();
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(end - begin).count();
                const double qps = elapsed_ms == 0.0
                    ? 0.0 : static_cast<double>(queries.count()) * 1000.0 / elapsed_ms;
                const double average_recall = queries.count() == 0
                    ? 0.0 : recall_sum / static_cast<double>(queries.count());
                const RunRow row{repeat, ef, elapsed_ms, average_recall, qps};
                run_rows.push_back(row);
                rows_by_ef[ef].push_back(row);
                std::cout << "  Lsearch=" << ef << ", efs=" << ef
                          << ", time=" << elapsed_ms
                          << "ms, avg_recall=" << average_recall << '\n';
                std::cout << "QPS = " << qps << '\n';
            }
        }

        std::ofstream details(result_dir / "search_time_details.csv");
        details << "Repeat,Lsearch,efs,Time_ms,Avg_Recall,QPS,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations\n";
        details << std::setprecision(10);
        for (const auto& row : run_rows) {
            details << row.repeat << ',' << row.ef << ',' << row.ef << ','
                    << row.elapsed_time_ms << ',' << row.recall << ',' << row.qps
                    << ",0,0,0\n";
        }

        std::ofstream query_details(
            result_dir / ("query_details_repeat" + std::to_string(num_repeats) + ".csv"));
        query_details << "QueryID,Lsearch,SearchTime_ms,Recall,Route,Selectivity,VisitedPoints,VisitedEdges,DistanceComputations\n";
        query_details << std::setprecision(10);
        for (const auto& row : query_rows) {
            query_details << row.query_id << ',' << row.ef << ',' << row.search_time_ms << ','
                          << row.recall << ",rwalks,-1,0,0,0\n";
        }

        std::ofstream summary(result_dir / "search_time_summary.csv");
        std::ofstream summary_qps(result_dir / "search_time_summary_qps.csv");
        summary << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations\n";
        summary_qps << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,QPS,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations\n";
        summary << std::setprecision(10);
        summary_qps << std::setprecision(10);
        for (const auto& [ef, rows] : rows_by_ef) {
            std::vector<double> times, recalls, qps_values;
            for (const auto& row : rows) {
                times.push_back(row.elapsed_time_ms);
                recalls.push_back(row.recall);
                qps_values.push_back(row.qps);
            }
            summary << ef << ',' << ef << ',' << rwalks_adapter::mean(times) << ','
                    << rwalks_adapter::mean(recalls) << ",0,0,0\n";
            summary_qps << ef << ',' << ef << ',' << rwalks_adapter::mean(times) << ','
                        << rwalks_adapter::mean(recalls) << ','
                        << rwalks_adapter::mean(qps_values) << ",0,0,0\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RWalks search failed: " << error.what() << '\n';
        return 1;
    }
}
