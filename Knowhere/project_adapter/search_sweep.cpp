#include "common.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <unordered_set>

#include <faiss/cppcontrib/knowhere/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/HnswSearcher.h>
#include <faiss/cppcontrib/knowhere/utils/Bitset.h>
#include <faiss/impl/DistanceComputer.h>
#include <knowhere/bitsetview.h>
#include <knowhere/bitsetview_idselector.h>
#include <omp.h>

namespace {

struct DummyVisitor {
    void visit_level(int) {}
    void visit_edge(int, std::int32_t, std::int32_t, float) {}
};

struct QueryFilter {
    std::vector<std::uint8_t> excluded;
    std::size_t eligible = 0;
};

struct QueryStats {
    double time_ms = 0.0;
    double recall = 0.0;
    std::size_t visited_points = 0;
    std::size_t visited_edges = 0;
    std::size_t distance_computations = 0;
    std::size_t eligible = 0;
    bool brute_force = false;
};

struct RunRow {
    int repeat = 0;
    int ef = 0;
    double time_ms = 0.0;
    double recall = 0.0;
    double qps = 0.0;
    double visited_points = 0.0;
    double visited_edges = 0.0;
    double distance_computations = 0.0;
    double brute_force_queries = 0.0;
};

std::vector<std::int32_t> load_groundtruth(
    const std::filesystem::path& path, std::size_t num_queries, int topk) {
    const std::size_t expected = num_queries * static_cast<std::size_t>(topk);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open ground truth: " + path.string());
    const std::streamsize size = input.tellg();
    if (size != static_cast<std::streamsize>(expected * sizeof(std::int32_t)))
        throw std::runtime_error("ground-truth size does not match query count and top-k");
    input.seekg(0, std::ios::beg);
    std::vector<std::int32_t> ids(expected);
    input.read(reinterpret_cast<char*>(ids.data()), size);
    if (!input) throw std::runtime_error("failed to read ground truth: " + path.string());
    return ids;
}

double recall_at_k(const std::int32_t* truth, const faiss::idx_t* result, int topk) {
    std::unordered_set<std::int32_t> valid;
    for (int i = 0; i < topk; ++i) if (truth[i] >= 0) valid.insert(truth[i]);
    if (valid.empty()) return 0.0;
    int correct = 0;
    for (int i = 0; i < topk; ++i)
        if (result[i] >= 0 && valid.count(static_cast<std::int32_t>(result[i]))) ++correct;
    return static_cast<double>(correct) / valid.size();
}

QueryFilter make_filter(
    const milvus_adapter::FilterIndex& index, const std::vector<int>& labels) {
    QueryFilter filter;
    filter.excluded.assign((index.vector_count + 7) / 8, 0xff);
    const auto eligible = milvus_adapter::eligible_ids(index, labels);
    filter.eligible = eligible.size();
    for (std::uint32_t id : eligible)
        filter.excluded[id >> 3] &= static_cast<std::uint8_t>(~(std::uint8_t(1) << (id & 7)));
    return filter;
}

std::unique_ptr<faiss::DistanceComputer> distance_computer(
    const milvus_adapter::kh::IndexHNSWFlat& index) {
    faiss::DistanceComputer* computer = index.storage->get_distance_computer();
    if (faiss::cppcontrib::knowhere::is_similarity_metric(index.metric_type))
        computer = new faiss::NegativeDistanceComputer(computer);
    return std::unique_ptr<faiss::DistanceComputer>(computer);
}

milvus_adapter::kh::HNSWStats graph_search(
    const milvus_adapter::kh::IndexHNSWFlat& index,
    const float* query,
    const QueryFilter& filter,
    int topk,
    int ef,
    float* distances,
    faiss::idx_t* labels) {
    auto computer = distance_computer(index);
    computer->set_query(query);
    milvus_adapter::kh::Bitset visited =
        milvus_adapter::kh::Bitset::create_cleared(static_cast<std::size_t>(index.ntotal));
    const std::size_t filtered = static_cast<std::size_t>(index.ntotal) - filter.eligible;
    const knowhere::BitsetView view(filter.excluded.data(), static_cast<std::size_t>(index.ntotal), filtered);
    const knowhere::BitsetViewIDSelector selector(view);
    milvus_adapter::kh::SearchParametersHNSW parameters;
    parameters.efSearch = ef;
    DummyVisitor visitor;
    milvus_adapter::kh::v2_hnsw_searcher<
        faiss::DistanceComputer, DummyVisitor, milvus_adapter::kh::Bitset,
        knowhere::BitsetViewIDSelector>
        searcher(index.hnsw, *computer, visitor, visited, selector,
                 view.filter_ratio() * 0.7f, &parameters);
    auto stats = searcher.search(topk, distances, labels);
    if (faiss::cppcontrib::knowhere::is_similarity_metric(index.metric_type))
        for (int i = 0; i < topk; ++i) distances[i] = -distances[i];
    return stats;
}

void brute_force_search(
    const milvus_adapter::kh::IndexHNSWFlat& index,
    const float* query,
    const QueryFilter& filter,
    int topk,
    float* distances,
    faiss::idx_t* labels) {
    auto computer = distance_computer(index);
    computer->set_query(query);
    using Candidate = std::pair<float, faiss::idx_t>;
    std::priority_queue<Candidate> heap;
    for (faiss::idx_t id = 0; id < index.ntotal; ++id) {
        if (filter.excluded[static_cast<std::size_t>(id) >> 3] &
            (std::uint8_t(1) << (static_cast<std::size_t>(id) & 7))) continue;
        const float distance = (*computer)(id);
        if (heap.size() < static_cast<std::size_t>(topk)) heap.emplace(distance, id);
        else if (distance < heap.top().first) {
            heap.pop();
            heap.emplace(distance, id);
        }
    }
    int position = static_cast<int>(heap.size());
    for (int i = position; i < topk; ++i) {
        distances[i] = std::numeric_limits<float>::max();
        labels[i] = -1;
    }
    while (!heap.empty()) {
        --position;
        distances[position] = heap.top().first;
        labels[position] = heap.top().second;
        heap.pop();
    }
    if (faiss::cppcontrib::knowhere::is_similarity_metric(index.metric_type))
        for (int i = 0; i < topk; ++i) distances[i] = -distances[i];
}

double mean(const std::vector<double>& values) {
    return values.empty() ? 0.0 : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 13) {
            std::cerr << "usage: " << argv[0]
                      << " <query.fvecs> <query_labels.txt> <gt_ids.bin> <index.bin> <filters.bin>"
                         " <result_dir> <topk> <num_repeats> <num_threads> <dist_fn>"
                         " <disable_fallback:0|1> <ef> [ef ...]\n";
            return 2;
        }
        const std::filesystem::path query_path = argv[1];
        const std::filesystem::path query_labels_path = argv[2];
        const std::filesystem::path groundtruth_path = argv[3];
        const std::filesystem::path index_path = argv[4];
        const std::filesystem::path filters_path = argv[5];
        const std::filesystem::path result_dir = argv[6];
        const int topk = std::stoi(argv[7]);
        const int repeats = std::stoi(argv[8]);
        const int num_threads = std::stoi(argv[9]);
        const std::string dist_fn = argv[10];
        const bool disable_fallback = std::stoi(argv[11]) != 0;
        if (topk <= 0 || repeats <= 0 || num_threads <= 0)
            throw std::runtime_error("top-k, repeats and num_threads must be positive");
        std::vector<int> ef_values;
        for (int i = 12; i < argc; ++i) {
            const int ef = std::stoi(argv[i]);
            if (ef < topk) throw std::runtime_error("every ef value must be >= top-k");
            ef_values.push_back(ef);
        }
        if (ef_values.empty()) throw std::runtime_error("at least one ef value is required");

        auto queries = milvus_adapter::load_fvecs(query_path);
        if (milvus_adapter::uses_cosine_normalization(dist_fn)) milvus_adapter::normalize_rows(queries);
        const auto query_labels = milvus_adapter::load_label_rows(query_labels_path);
        if (query_labels.size() != queries.count)
            throw std::runtime_error("query-label row count does not match query vectors");
        const auto truth = load_groundtruth(groundtruth_path, queries.count, topk);
        auto index = milvus_adapter::load_index(index_path);
        const auto filters = milvus_adapter::load_filter_index(filters_path);
        if (index->d != queries.dimension) throw std::runtime_error("query dimension does not match index");
        if (index->metric_type != milvus_adapter::parse_metric(dist_fn))
            throw std::runtime_error("configured metric does not match index");
        if (filters.vector_count != static_cast<std::size_t>(index->ntotal))
            throw std::runtime_error("filter index vector count does not match HNSW index");

        omp_set_num_threads(num_threads);
        std::filesystem::create_directories(result_dir);
        std::vector<RunRow> runs;
        std::map<int, std::vector<RunRow>> by_ef;
        std::ofstream query_output(result_dir / ("query_details_repeat" + std::to_string(repeats) + ".csv"));
        query_output << "Repeat,QueryID,Lsearch,SearchTime_ms,Recall,Route,Selectivity,VisitedPoints,VisitedEdges,DistanceComputations\n";
        query_output << std::setprecision(10);

        std::vector<faiss::idx_t> result_ids(queries.count * static_cast<std::size_t>(topk));
        std::vector<float> distances(result_ids.size());
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (int ef : ef_values) {
                std::vector<QueryStats> per_query(queries.count);
                const auto batch_begin = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
                for (std::int64_t query_id = 0; query_id < static_cast<std::int64_t>(queries.count); ++query_id) {
                    const auto begin = std::chrono::steady_clock::now();
                    const auto filter = make_filter(
                        filters, query_labels[static_cast<std::size_t>(query_id)]);
                    QueryStats& stats = per_query[static_cast<std::size_t>(query_id)];
                    stats.eligible = filter.eligible;
                    const bool fallback = !disable_fallback &&
                        (filter.eligible * 100 <= static_cast<std::size_t>(index->ntotal) * 7 ||
                         static_cast<std::size_t>(topk) * 2 >= filter.eligible ||
                         static_cast<std::size_t>(topk) * 2 >= static_cast<std::size_t>(index->ntotal));
                    stats.brute_force = fallback;
                    float* query_distances = distances.data() + query_id * topk;
                    faiss::idx_t* query_ids = result_ids.data() + query_id * topk;
                    if (fallback) {
                        brute_force_search(*index, queries.values.data() + query_id * queries.dimension,
                                           filter, topk, query_distances, query_ids);
                        stats.visited_points = filter.eligible;
                        stats.distance_computations = filter.eligible;
                    } else {
                        const auto hnsw_stats = graph_search(
                            *index, queries.values.data() + query_id * queries.dimension,
                            filter, topk, ef, query_distances, query_ids);
                        stats.visited_points = hnsw_stats.ndis;
                        stats.visited_edges = hnsw_stats.nhops;
                        stats.distance_computations = hnsw_stats.ndis;
                    }
                    stats.time_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin).count();
                    stats.recall = recall_at_k(truth.data() + query_id * topk, query_ids, topk);
                }
                const double batch_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - batch_begin).count();
                double recall_sum = 0.0, points = 0.0, edges = 0.0, computations = 0.0, bf = 0.0;
                for (std::size_t query_id = 0; query_id < queries.count; ++query_id) {
                    const auto& stats = per_query[query_id];
                    recall_sum += stats.recall;
                    points += stats.visited_points;
                    edges += stats.visited_edges;
                    computations += stats.distance_computations;
                    bf += stats.brute_force ? 1.0 : 0.0;
                    const double selectivity = index->ntotal == 0 ? 0.0 :
                        static_cast<double>(stats.eligible) / static_cast<double>(index->ntotal);
                    query_output << repeat << ',' << query_id << ',' << ef << ',' << stats.time_ms << ','
                                 << stats.recall << ',' << (stats.brute_force ? "bruteforce" : "hnsw") << ','
                                 << selectivity << ',' << stats.visited_points << ',' << stats.visited_edges << ','
                                 << stats.distance_computations << '\n';
                }
                const double count = static_cast<double>(queries.count);
                const RunRow row{repeat, ef, batch_ms, count ? recall_sum / count : 0.0,
                    batch_ms > 0.0 ? count * 1000.0 / batch_ms : 0.0,
                    count ? points / count : 0.0, count ? edges / count : 0.0,
                    count ? computations / count : 0.0, bf};
                runs.push_back(row);
                by_ef[ef].push_back(row);
                std::cout << "Lsearch=" << ef << " time_ms=" << batch_ms
                          << " avg_recall=" << row.recall << " QPS=" << row.qps
                          << " brute_force_queries=" << bf << '\n';
            }
        }

        std::ofstream details(result_dir / "search_time_details.csv");
        details << "Repeat,Lsearch,efs,Time_ms,Avg_Recall,QPS,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations,BruteForceQueries\n";
        details << std::setprecision(10);
        for (const auto& row : runs)
            details << row.repeat << ',' << row.ef << ',' << row.ef << ',' << row.time_ms << ','
                    << row.recall << ',' << row.qps << ',' << row.visited_points << ',' << row.visited_edges << ','
                    << row.distance_computations << ',' << row.brute_force_queries << '\n';

        std::ofstream summary(result_dir / "search_time_summary.csv");
        std::ofstream summary_qps(result_dir / "search_time_summary_qps.csv");
        summary << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations,Average_BruteForceQueries\n";
        summary_qps << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall,QPS,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations,Average_BruteForceQueries\n";
        summary << std::setprecision(10);
        summary_qps << std::setprecision(10);
        for (const auto& [ef, rows] : by_ef) {
            std::vector<double> times, recalls, qps, points, edges, computations, bfs;
            for (const auto& row : rows) {
                times.push_back(row.time_ms); recalls.push_back(row.recall); qps.push_back(row.qps);
                points.push_back(row.visited_points); edges.push_back(row.visited_edges);
                computations.push_back(row.distance_computations); bfs.push_back(row.brute_force_queries);
            }
            summary << ef << ',' << ef << ',' << mean(times) << ',' << mean(recalls) << ','
                    << mean(points) << ',' << mean(edges) << ',' << mean(computations) << ',' << mean(bfs) << '\n';
            summary_qps << ef << ',' << ef << ',' << mean(times) << ',' << mean(recalls) << ',' << mean(qps) << ','
                        << mean(points) << ',' << mean(edges) << ',' << mean(computations) << ',' << mean(bfs) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Milvus HNSW search failed: " << error.what() << '\n';
        return 1;
    }
}
