#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/index_io.h>

namespace milvus_adapter {

namespace kh = faiss::cppcontrib::knowhere;

struct FvecsData {
    std::size_t count = 0;
    int dimension = 0;
    std::vector<float> values;
};

inline FvecsData load_fvecs(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open fvecs file: " + path.string());
    const std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::int32_t dimension = 0;
    input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    if (!input || dimension <= 0) throw std::runtime_error("invalid fvecs file: " + path.string());
    const std::uint64_t record_bytes = sizeof(std::int32_t) +
        static_cast<std::uint64_t>(dimension) * sizeof(float);
    if (static_cast<std::uint64_t>(file_size) % record_bytes != 0)
        throw std::runtime_error("fvecs size is not a whole number of records: " + path.string());
    FvecsData result;
    result.dimension = dimension;
    result.count = static_cast<std::size_t>(static_cast<std::uint64_t>(file_size) / record_bytes);
    result.values.resize(result.count * static_cast<std::size_t>(dimension));
    input.seekg(0, std::ios::beg);
    for (std::size_t row = 0; row < result.count; ++row) {
        std::int32_t row_dimension = 0;
        input.read(reinterpret_cast<char*>(&row_dimension), sizeof(row_dimension));
        if (!input || row_dimension != dimension)
            throw std::runtime_error("inconsistent fvecs dimension at row " + std::to_string(row));
        input.read(reinterpret_cast<char*>(result.values.data() + row * dimension),
                   static_cast<std::streamsize>(dimension) * sizeof(float));
        if (!input) throw std::runtime_error("truncated fvecs payload at row " + std::to_string(row));
    }
    return result;
}

inline std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline faiss::MetricType parse_metric(const std::string& name) {
    const std::string metric = lower_copy(name);
    if (metric == "l2" || metric == "euclidean") return faiss::METRIC_L2;
    if (metric == "ip" || metric == "inner_product" || metric == "inner-product" || metric == "cosine")
        return faiss::METRIC_INNER_PRODUCT;
    throw std::runtime_error("unsupported distance function: " + name);
}

inline bool uses_cosine_normalization(const std::string& name) {
    return lower_copy(name) == "cosine";
}

inline void normalize_rows(FvecsData& data) {
    for (std::size_t row = 0; row < data.count; ++row) {
        float* values = data.values.data() + row * data.dimension;
        double squared_norm = 0.0;
        for (int column = 0; column < data.dimension; ++column)
            squared_norm += static_cast<double>(values[column]) * values[column];
        if (squared_norm == 0.0) continue;
        const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
        for (int column = 0; column < data.dimension; ++column) values[column] *= inverse_norm;
    }
}

inline std::vector<int> parse_label_line(std::string line) {
    for (char& c : line) if (c == ',') c = ' ';
    std::vector<int> labels;
    std::size_t position = 0;
    while (position < line.size()) {
        while (position < line.size() && std::isspace(static_cast<unsigned char>(line[position]))) ++position;
        if (position == line.size()) break;
        std::size_t consumed = 0;
        labels.push_back(std::stoi(line.substr(position), &consumed));
        position += consumed;
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    return labels;
}

inline std::vector<std::vector<int>> load_label_rows(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open label file: " + path.string());
    std::vector<std::vector<int>> rows;
    std::string line;
    while (std::getline(input, line)) rows.push_back(parse_label_line(line));
    return rows;
}

template <typename ToGraph, typename FromGraph>
inline void copy_graph(ToGraph& to, const FromGraph& from) {
    to.assign_probas = from.assign_probas;
    to.cum_nneighbor_per_level = from.cum_nneighbor_per_level;
    to.levels = from.levels;
    to.offsets = from.offsets;
    to.neighbors = from.neighbors;
    to.entry_point = from.entry_point;
    to.max_level = from.max_level;
    to.efConstruction = from.efConstruction;
    to.efSearch = from.efSearch;
    to.check_relative_distance = from.check_relative_distance;
    to.search_bounded_queue = from.search_bounded_queue;
}

inline void save_index(const kh::IndexHNSWFlat& source, const std::filesystem::path& path) {
    const int M = source.hnsw.nb_neighbors(0) / 2;
    faiss::IndexHNSWFlat persisted(source.d, M, source.metric_type);
    delete persisted.storage;
    persisted.storage = source.storage;
    persisted.own_fields = false;
    persisted.ntotal = source.ntotal;
    persisted.is_trained = source.is_trained;
    persisted.metric_arg = source.metric_arg;
    copy_graph(persisted.hnsw, source.hnsw);
    std::filesystem::create_directories(path.parent_path());
    faiss::write_index(&persisted, path.c_str());
    persisted.storage = nullptr;
}

inline std::unique_ptr<kh::IndexHNSWFlat> load_index(const std::filesystem::path& path) {
    std::unique_ptr<faiss::Index> raw(faiss::read_index(path.c_str()));
    auto* persisted = dynamic_cast<faiss::IndexHNSWFlat*>(raw.get());
    if (!persisted) throw std::runtime_error("not a FAISS HNSWFlat persistence container: " + path.string());
    const int M = persisted->hnsw.nb_neighbors(0) / 2;
    auto result = std::make_unique<kh::IndexHNSWFlat>(persisted->d, M, persisted->metric_type);
    delete result->storage;
    result->storage = persisted->storage;
    result->own_fields = true;
    persisted->storage = nullptr;
    persisted->own_fields = false;
    result->ntotal = persisted->ntotal;
    result->is_trained = persisted->is_trained;
    result->metric_arg = persisted->metric_arg;
    copy_graph(result->hnsw, persisted->hnsw);
    return result;
}

struct FilterIndex {
    std::uint64_t vector_count = 0;
    std::unordered_map<int, std::vector<std::uint32_t>> postings;
};

inline FilterIndex build_filter_index(const std::vector<std::vector<int>>& rows) {
    FilterIndex result;
    result.vector_count = rows.size();
    for (std::size_t id = 0; id < rows.size(); ++id)
        for (int label : rows[id]) result.postings[label].push_back(static_cast<std::uint32_t>(id));
    return result;
}

inline void save_filter_index(const FilterIndex& index, const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create filter index: " + path.string());
    const std::uint64_t magic = 0x31544c46484dULL;  // MHFILT1
    const std::uint64_t label_count = index.postings.size();
    output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    output.write(reinterpret_cast<const char*>(&index.vector_count), sizeof(index.vector_count));
    output.write(reinterpret_cast<const char*>(&label_count), sizeof(label_count));
    std::vector<int> labels;
    labels.reserve(index.postings.size());
    for (const auto& item : index.postings) labels.push_back(item.first);
    std::sort(labels.begin(), labels.end());
    for (int label : labels) {
        const auto& ids = index.postings.at(label);
        const std::int64_t stored_label = label;
        const std::uint64_t count = ids.size();
        output.write(reinterpret_cast<const char*>(&stored_label), sizeof(stored_label));
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        output.write(reinterpret_cast<const char*>(ids.data()),
                     static_cast<std::streamsize>(ids.size() * sizeof(ids[0])));
    }
    if (!output) throw std::runtime_error("failed to write filter index: " + path.string());
}

inline FilterIndex load_filter_index(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open filter index: " + path.string());
    std::uint64_t magic = 0, label_count = 0;
    FilterIndex result;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char*>(&result.vector_count), sizeof(result.vector_count));
    input.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
    if (!input || magic != 0x31544c46484dULL) throw std::runtime_error("invalid filter index: " + path.string());
    for (std::uint64_t item = 0; item < label_count; ++item) {
        std::int64_t label = 0;
        std::uint64_t count = 0;
        input.read(reinterpret_cast<char*>(&label), sizeof(label));
        input.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (count > result.vector_count) throw std::runtime_error("invalid posting count in: " + path.string());
        auto& ids = result.postings[static_cast<int>(label)];
        ids.resize(static_cast<std::size_t>(count));
        input.read(reinterpret_cast<char*>(ids.data()),
                   static_cast<std::streamsize>(ids.size() * sizeof(ids[0])));
    }
    if (!input) throw std::runtime_error("truncated filter index: " + path.string());
    return result;
}

inline std::vector<std::uint32_t> eligible_ids(const FilterIndex& index, const std::vector<int>& labels) {
    if (labels.empty()) {
        std::vector<std::uint32_t> all(index.vector_count);
        for (std::size_t id = 0; id < all.size(); ++id) all[id] = static_cast<std::uint32_t>(id);
        return all;
    }
    const auto first = index.postings.find(labels.front());
    if (first == index.postings.end()) return {};
    std::vector<std::uint32_t> result = first->second;
    for (std::size_t i = 1; i < labels.size() && !result.empty(); ++i) {
        const auto found = index.postings.find(labels[i]);
        if (found == index.postings.end()) return {};
        std::vector<std::uint32_t> intersection;
        intersection.reserve(std::min(result.size(), found->second.size()));
        std::set_intersection(result.begin(), result.end(), found->second.begin(), found->second.end(),
                              std::back_inserter(intersection));
        result.swap(intersection);
    }
    return result;
}

inline std::size_t filter_memory_bytes(const FilterIndex& index) {
    std::size_t bytes = sizeof(FilterIndex) + index.postings.bucket_count() * sizeof(void*);
    for (const auto& [label, ids] : index.postings)
        bytes += sizeof(decltype(index.postings)::value_type) + 2 * sizeof(void*) +
            ids.capacity() * sizeof(std::uint32_t);
    return bytes;
}

}  // namespace milvus_adapter
