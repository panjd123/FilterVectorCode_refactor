#include "common.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <unordered_set>

#include <faiss/IndexPathSeer.h>
#include <faiss/index_io.h>
#include <omp.h>

namespace {

class ContainmentSelector final : public faiss::IDSelector {
public:
    ContainmentSelector(
        const std::vector<std::vector<int>>& base_labels,
        const std::vector<std::vector<int>>& query_labels)
        : base_labels_(base_labels), query_labels_(query_labels) {}

    bool is_member(faiss::idx_t id) const override {
        return id >= 0 && static_cast<std::size_t>(id) < base_labels_.size() &&
            project_adapter::contains_all(base_labels_[id], query_labels_.front());
    }

    bool is_member(faiss::idx_t id, int query_id) const override {
        return id >= 0 && query_id >= 0 &&
            static_cast<std::size_t>(id) < base_labels_.size() &&
            static_cast<std::size_t>(query_id) < query_labels_.size() &&
            project_adapter::contains_all(base_labels_[id], query_labels_[query_id]);
    }

private:
    const std::vector<std::vector<int>>& base_labels_;
    const std::vector<std::vector<int>>& query_labels_;
};

std::vector<std::int32_t> load_groundtruth(
    const std::filesystem::path& path,
    std::size_t num_queries,
    int topk) {
    const std::size_t expected = num_queries * static_cast<std::size_t>(topk);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open ground truth: " + path.string());
    }
    const std::streamsize file_size = input.tellg();
    if (file_size != static_cast<std::streamsize>(expected * sizeof(std::int32_t))) {
        throw std::runtime_error("ground-truth size does not match query count and top-k");
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::int32_t> ids(expected);
    input.read(reinterpret_cast<char*>(ids.data()), file_size);
    if (!input) {
        throw std::runtime_error("failed to read ground truth: " + path.string());
    }
    return ids;
}

double query_recall(const std::int32_t* groundtruth, const faiss::idx_t* result, int topk) {
    std::unordered_set<std::int32_t> valid;
    for (int i = 0; i < topk; ++i) {
        if (groundtruth[i] >= 0) {
            valid.insert(groundtruth[i]);
        }
    }
    if (valid.empty()) {
        return 0.0;
    }
    int correct = 0;
    for (int i = 0; i < topk; ++i) {
        if (result[i] >= 0 && valid.count(static_cast<std::int32_t>(result[i])) != 0) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / valid.size();
}

struct RunRow {
    int repeat;
    int ef;
    double time_ms;
    double recall;
    double qps;
    double visited_points;
    double visited_edges;
    double distance_computations;
};

struct QueryRow {
    std::size_t query_id;
    int ef;
    double time_ms;
    double recall;
    double visited_points;
    double visited_edges;
    double distance_computations;
};

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 16) {
            std::cerr << "usage: " << argv[0]
                      << " <query.fvecs> <base_labels.txt> <query_labels.txt> <gt_ids.bin>"
                         " <index.bin> <result_dir> <topk> <num_repeats> <num_threads>"
                         " <dist_fn> <M_exp_search> <virtual_overhead> <dist_comp_ns> <filter_ns> <ef> [ef ...]\n";
            return 2;
        }

        const std::filesystem::path query_path = argv[1];
        const std::filesystem::path base_label_path = argv[2];
        const std::filesystem::path query_label_path = argv[3];
        const std::filesystem::path groundtruth_path = argv[4];
        const std::filesystem::path index_path = argv[5];
        const std::filesystem::path result_dir = argv[6];
        const int topk = std::stoi(argv[7]);
        const int num_repeats = std::stoi(argv[8]);
        const int num_threads = std::stoi(argv[9]);
        const std::string dist_fn = argv[10];
        const int M_exp_search = std::stoi(argv[11]);
        const int virtual_overhead = std::stoi(argv[12]);
        const long long dist_comp_ns = std::stoll(argv[13]);
        const long long filter_ns = std::stoll(argv[14]);
        if (topk <= 0 || num_repeats <= 0 || num_threads <= 0 || M_exp_search < 0) {
            throw std::runtime_error("top-k, repeats, and threads must be positive; M_exp_search must be nonnegative");
        }
        if (M_exp_search == 0 &&
            (virtual_overhead <= 0 || dist_comp_ns <= 0 || filter_ns <= 0)) {
            throw std::runtime_error(
                "full-tech mode (M_exp_search=0) requires positive virtual_overhead, dist_comp_ns, and filter_ns");
        }
        std::vector<int> ef_values;
        for (int argument = 15; argument < argc; ++argument) {
            const int ef = std::stoi(argv[argument]);
            if (ef <= 0) {
                throw std::runtime_error("ef/Lsearch values must be positive");
            }
            ef_values.push_back(ef);
        }

        project_adapter::FvecsData queries = project_adapter::load_fvecs(query_path);
        if (project_adapter::uses_cosine_normalization(dist_fn)) {
            project_adapter::normalize_rows(queries);
        }
        const auto base_labels = project_adapter::load_label_rows(base_label_path);
        const auto query_labels = project_adapter::load_label_rows(query_label_path);
        if (query_labels.size() != queries.count) {
            throw std::runtime_error("query-label row count does not match query vectors");
        }
        const auto groundtruth = load_groundtruth(groundtruth_path, queries.count, topk);

        std::unique_ptr<faiss::Index> raw_index(faiss::read_index(index_path.c_str()));
        auto* index = dynamic_cast<faiss::IndexPathSeerFlat*>(raw_index.get());
        if (!index) {
            throw std::runtime_error("index is not an IndexPathSeerFlat: " + index_path.string());
        }
        if (index->metric_type != project_adapter::parse_metric(dist_fn)) {
            throw std::runtime_error("configured distance function does not match PathSeer index metric");
        }
        if (index->d != queries.dimension) {
            throw std::runtime_error("query dimension does not match PathSeer index");
        }
        if (base_labels.size() != static_cast<std::size_t>(index->ntotal)) {
            throw std::runtime_error("base-label row count does not match PathSeer index size");
        }
        if (M_exp_search > index->pathseer.M_exp) {
            throw std::runtime_error("M_exp_search exceeds the index M_exp construction value");
        }
        if (M_exp_search == 0) {
            index->set_profiled_dist_comp_ns(dist_comp_ns);
            index->set_profiled_filter_ns(filter_ns);
        }

        omp_set_num_threads(num_threads);
        ContainmentSelector selector(base_labels, query_labels);
        std::filesystem::create_directories(result_dir);

        std::vector<RunRow> run_rows;
        std::vector<QueryRow> query_rows;
        std::map<int, std::vector<RunRow>> rows_by_ef;
        std::vector<faiss::idx_t> result_ids(queries.count * static_cast<std::size_t>(topk));
        std::vector<float> distances(result_ids.size());

        for (int repeat = 0; repeat < num_repeats; ++repeat) {
            std::cout << "\n=== Repeat " << (repeat + 1) << '/' << num_repeats << " ===\n";
            for (int ef : ef_values) {
                faiss::SearchParametersPathSeer parameters;
                parameters.sel = &selector;
                parameters.efSearch = ef;
                parameters.M_exp_search = M_exp_search;
                parameters.virtual_overhead = virtual_overhead;
                parameters.check_relative_distance = true;
                faiss::pathseer_stats.reset();

                const auto begin = std::chrono::steady_clock::now();
                index->search(
                    static_cast<faiss::idx_t>(queries.count), queries.values.data(), topk,
                    distances.data(), result_ids.data(), &parameters);
                const auto end = std::chrono::steady_clock::now();
                const double time_ms =
                    std::chrono::duration<double, std::milli>(end - begin).count();
                const double qps = time_ms > 0.0 ? queries.count * 1000.0 / time_ms : 0.0;

                double recall_sum = 0.0;
                std::vector<double> recalls(queries.count);
                for (std::size_t query_id = 0; query_id < queries.count; ++query_id) {
                    recalls[query_id] = query_recall(
                        groundtruth.data() + query_id * topk,
                        result_ids.data() + query_id * topk,
                        topk);
                    recall_sum += recalls[query_id];
                }
                const double query_count = static_cast<double>(queries.count);
                const double avg_recall = queries.count == 0 ? 0.0 : recall_sum / query_count;
                const double visited_points = queries.count == 0 ? 0.0 :
                    static_cast<double>(faiss::pathseer_stats.nfilter) / query_count;
                const double visited_edges = queries.count == 0 ? 0.0 :
                    static_cast<double>(faiss::pathseer_stats.nhop) / query_count;
                const double distance_computations = queries.count == 0 ? 0.0 :
                    static_cast<double>(faiss::pathseer_stats.ndis) / query_count;

                const RunRow row{repeat, ef, time_ms, avg_recall, qps, visited_points,
                    visited_edges, distance_computations};
                run_rows.push_back(row);
                rows_by_ef[ef].push_back(row);
                const double average_query_time = queries.count == 0 ? 0.0 : time_ms / query_count;
                for (std::size_t query_id = 0; query_id < queries.count; ++query_id) {
                    query_rows.push_back({query_id, ef, average_query_time, recalls[query_id],
                        visited_points, visited_edges, distance_computations});
                }

                std::cout << "  Lsearch=" << ef << ", efs=" << ef << ", time=" << time_ms
                          << "ms, avg_recall=" << avg_recall << '\n';
                std::cout << "QPS = " << qps << '\n';
            }
        }

        std::ofstream details(result_dir / "search_time_details.csv");
        details << "Repeat,Lsearch,efs,Time_ms,Avg_Recall,QPS,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations\n";
        details << std::setprecision(10);
        for (const RunRow& row : run_rows) {
            details << row.repeat << ',' << row.ef << ',' << row.ef << ',' << row.time_ms << ','
                    << row.recall << ',' << row.qps << ',' << row.visited_points << ','
                    << row.visited_edges << ',' << row.distance_computations << '\n';
        }

        std::ofstream query_details(
            result_dir / ("query_details_repeat" + std::to_string(num_repeats) + ".csv"));
        query_details << "QueryID,Lsearch,SearchTime_ms,Recall,Route,Selectivity,VisitedPoints,VisitedEdges,DistanceComputations\n";
        query_details << std::setprecision(10);
        for (const QueryRow& row : query_rows) {
            query_details << row.query_id << ',' << row.ef << ',' << row.time_ms << ','
                          << row.recall << ",graph,-1," << row.visited_points << ','
                          << row.visited_edges << ',' << row.distance_computations << '\n';
        }

        std::ofstream summary(result_dir / "search_time_summary.csv");
        std::ofstream summary_qps(result_dir / "search_time_summary_qps.csv");
        summary << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations\n";
        summary_qps << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,QPS,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations\n";
        summary << std::setprecision(10);
        summary_qps << std::setprecision(10);
        for (const auto& [ef, rows] : rows_by_ef) {
            std::vector<double> times, recalls, qps_values, points, edges, distance_values;
            for (const RunRow& row : rows) {
                times.push_back(row.time_ms);
                recalls.push_back(row.recall);
                qps_values.push_back(row.qps);
                points.push_back(row.visited_points);
                edges.push_back(row.visited_edges);
                distance_values.push_back(row.distance_computations);
            }
            summary << ef << ',' << ef << ',' << mean(times) << ',' << mean(recalls) << ','
                    << mean(points) << ',' << mean(edges) << ',' << mean(distance_values) << '\n';
            summary_qps << ef << ',' << ef << ',' << mean(times) << ',' << mean(recalls) << ','
                        << mean(qps_values) << ',' << mean(points) << ',' << mean(edges) << ','
                        << mean(distance_values) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PathSeer search failed: " << error.what() << '\n';
        return 1;
    }
}
