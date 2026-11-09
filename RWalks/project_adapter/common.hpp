#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <omp.h>

#include <hnswlib/hnswlib.h>

namespace rwalks_adapter {

class MappedFvecs {
public:
    explicit MappedFvecs(const std::filesystem::path& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open fvecs file: " + path.string());
        }
        struct stat file_stat {};
        if (::fstat(fd_, &file_stat) != 0 || file_stat.st_size < 4) {
            close_fd();
            throw std::runtime_error("empty or invalid fvecs file: " + path.string());
        }
        size_ = static_cast<std::size_t>(file_stat.st_size);
        mapping_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            close_fd();
            throw std::runtime_error("mmap failed for fvecs file: " + path.string());
        }
        std::memcpy(&dimension_, mapping_, sizeof(dimension_));
        if (dimension_ <= 0) {
            throw std::runtime_error("invalid fvecs dimension in: " + path.string());
        }
        record_bytes_ = sizeof(std::int32_t) + static_cast<std::size_t>(dimension_) * sizeof(float);
        if (size_ % record_bytes_ != 0) {
            throw std::runtime_error("fvecs size is not a whole number of records: " + path.string());
        }
        count_ = size_ / record_bytes_;
        for (std::size_t row = 0; row < count_; ++row) {
            std::int32_t row_dimension = 0;
            std::memcpy(&row_dimension, bytes() + row * record_bytes_, sizeof(row_dimension));
            if (row_dimension != dimension_) {
                throw std::runtime_error("inconsistent fvecs dimension at row " + std::to_string(row));
            }
        }
    }

    MappedFvecs(const MappedFvecs&) = delete;
    MappedFvecs& operator=(const MappedFvecs&) = delete;

    ~MappedFvecs() {
        if (mapping_ != nullptr) {
            ::munmap(mapping_, size_);
        }
        close_fd();
    }

    std::size_t count() const { return count_; }
    int dimension() const { return dimension_; }

    const float* row(std::size_t index) const {
        return reinterpret_cast<const float*>(
            bytes() + index * record_bytes_ + sizeof(std::int32_t));
    }

private:
    const char* bytes() const { return static_cast<const char*>(mapping_); }
    void close_fd() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    std::filesystem::path path_;
    int fd_ = -1;
    void* mapping_ = nullptr;
    std::size_t size_ = 0;
    std::size_t record_bytes_ = 0;
    std::size_t count_ = 0;
    std::int32_t dimension_ = 0;
};

inline std::vector<int> parse_label_line(std::string line) {
    for (char& character : line) {
        if (character == ',') character = ' ';
    }
    std::vector<int> labels;
    std::size_t position = 0;
    while (position < line.size()) {
        while (position < line.size() &&
               std::isspace(static_cast<unsigned char>(line[position]))) ++position;
        if (position == line.size()) break;
        std::size_t consumed = 0;
        labels.push_back(std::stoi(line.substr(position), &consumed));
        position += consumed;
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    return labels;
}

inline std::vector<int> load_schema(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open label schema: " + path.string());
    std::vector<int> labels;
    std::string line;
    while (std::getline(input, line)) {
        const auto row = parse_label_line(line);
        labels.insert(labels.end(), row.begin(), row.end());
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    if (labels.empty()) throw std::runtime_error("label schema must not be empty: " + path.string());
    return labels;
}

inline std::unordered_map<int, std::size_t> schema_lookup(const std::vector<int>& schema) {
    std::unordered_map<int, std::size_t> result;
    result.reserve(schema.size() * 2);
    for (std::size_t index = 0; index < schema.size(); ++index) result.emplace(schema[index], index);
    return result;
}

inline std::vector<int> load_dense_labels(
    const std::filesystem::path& path,
    std::size_t expected_rows,
    const std::vector<int>& schema,
    bool reject_unknown) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open label file: " + path.string());
    const auto lookup = schema_lookup(schema);
    std::vector<int> dense(expected_rows * schema.size(), 0);
    std::string line;
    std::size_t row_id = 0;
    while (std::getline(input, line)) {
        if (row_id >= expected_rows) {
            throw std::runtime_error("label file has more rows than vectors: " + path.string());
        }
        for (int label : parse_label_line(line)) {
            const auto found = lookup.find(label);
            if (found == lookup.end()) {
                if (reject_unknown) {
                    throw std::runtime_error(
                        "query label " + std::to_string(label) + " is absent from the index schema");
                }
                continue;
            }
            dense[row_id * schema.size() + found->second] = 1;
        }
        ++row_id;
    }
    if (row_id != expected_rows) {
        throw std::runtime_error(
            "label row count " + std::to_string(row_id) + " does not match vector count " +
            std::to_string(expected_rows) + ": " + path.string());
    }
    return dense;
}

inline std::vector<std::int32_t> load_groundtruth(
    const std::filesystem::path& path,
    std::size_t query_count,
    int topk) {
    const std::size_t expected = query_count * static_cast<std::size_t>(topk);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open ground truth: " + path.string());
    const auto size = input.tellg();
    if (size != static_cast<std::streamsize>(expected * sizeof(std::int32_t))) {
        throw std::runtime_error("ground-truth size does not match query count and top-k");
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::int32_t> values(expected);
    input.read(reinterpret_cast<char*>(values.data()), size);
    if (!input) throw std::runtime_error("failed to read ground truth: " + path.string());
    return values;
}

inline double recall_at_k(
    const std::int32_t* groundtruth,
    const hnswlib::labeltype* result,
    int topk) {
    std::unordered_set<std::int32_t> truth;
    for (int index = 0; index < topk; ++index) {
        if (groundtruth[index] >= 0) truth.insert(groundtruth[index]);
    }
    if (truth.empty()) return 0.0;
    int correct = 0;
    for (int index = 0; index < topk; ++index) {
        if (result[index] != static_cast<hnswlib::labeltype>(-1) &&
            truth.count(static_cast<std::int32_t>(result[index])) != 0) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(truth.size());
}

inline bool already_visited(const std::vector<hnswlib::tableint>& visited, hnswlib::tableint id) {
    return std::find(visited.begin(), visited.end(), id) != visited.end();
}

inline void compute_random_walk_attributes(
    hnswlib::HierarchicalNSW<float>& index,
    int walk_length,
    int walk_count,
    float decay_factor,
    int num_threads,
    std::uint32_t random_seed) {
    if (walk_length <= 0 || walk_count <= 0 || decay_factor < 0.0f) {
        throw std::runtime_error("random-walk depth/count must be positive and alpha nonnegative");
    }
    const std::size_t dimensions = index.size_data_attr_per_element_ / sizeof(int);
    const std::size_t node_count = index.cur_element_count.load();
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        std::vector<float> aggregate(dimensions);
        std::vector<hnswlib::tableint> visited;
        std::vector<hnswlib::tableint> candidates;
        visited.reserve(static_cast<std::size_t>(walk_length));

#pragma omp for schedule(dynamic, 64)
        for (std::int64_t signed_node = 0;
             signed_node < static_cast<std::int64_t>(node_count); ++signed_node) {
            const auto node = static_cast<hnswlib::tableint>(signed_node);
            std::fill(aggregate.begin(), aggregate.end(), 0.0f);
            std::mt19937 generator(random_seed ^ (0x9e3779b9U * (node + 1U)));

            for (int walk_id = 0; walk_id < walk_count; ++walk_id) {
                hnswlib::tableint current = node;
                visited.clear();
                for (int step = 0; step < walk_length; ++step) {
                    auto* links = index.get_linklist0(current);
                    const std::size_t degree = index.getListCount(links);
                    if (degree == 0) break;
                    auto* neighbors = reinterpret_cast<hnswlib::tableint*>(links + 1);
                    candidates.clear();
                    candidates.reserve(degree);
                    for (std::size_t offset = 0; offset < degree; ++offset) {
                        if (!already_visited(visited, neighbors[offset])) candidates.push_back(neighbors[offset]);
                    }
                    if (candidates.empty()) {
                        visited.clear();
                        candidates.assign(neighbors, neighbors + degree);
                    }
                    std::uniform_int_distribution<std::size_t> distribution(0, candidates.size() - 1);
                    current = candidates[distribution(generator)];
                    visited.push_back(current);

                    const auto* attributes = reinterpret_cast<const int*>(
                        index.getDataAttrByInternalId(current));
                    const float weight = std::pow(decay_factor, static_cast<float>(step)) /
                        static_cast<float>(walk_length * walk_count);
                    for (std::size_t attribute = 0; attribute < dimensions; ++attribute) {
                        if (attributes[attribute] != 0) aggregate[attribute] += weight;
                    }
                }
            }

            const auto* own_attributes = reinterpret_cast<const int*>(
                index.getDataAttrByInternalId(node));
            for (std::size_t attribute = 0; attribute < dimensions; ++attribute) {
                if (own_attributes[attribute] != 0) aggregate[attribute] = 1.0f;
            }
            std::memcpy(index.getDataAttrAggByInternalId(node), aggregate.data(), dimensions * sizeof(float));
        }
    }

    // Match the upstream binding's normalization.
    std::vector<double> sums(dimensions, 0.0);
    std::vector<std::size_t> counts(dimensions, 0);
    std::vector<float> maxima(dimensions, 0.0f);
    for (std::size_t node = 0; node < node_count; ++node) {
        const auto* values = reinterpret_cast<const float*>(index.getDataAttrAggByInternalId(node));
        for (std::size_t attribute = 0; attribute < dimensions; ++attribute) {
            if (values[attribute] != 1.0f) {
                sums[attribute] += values[attribute];
                ++counts[attribute];
                maxima[attribute] = std::max(maxima[attribute], values[attribute]);
            }
        }
    }
    std::vector<float> means(dimensions, 0.0f);
    for (std::size_t attribute = 0; attribute < dimensions; ++attribute) {
        if (counts[attribute] != 0) {
            means[attribute] = static_cast<float>(sums[attribute] / counts[attribute]);
            maxima[attribute] -= means[attribute];
        }
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t signed_node = 0;
         signed_node < static_cast<std::int64_t>(node_count); ++signed_node) {
        auto* values = reinterpret_cast<float*>(
            index.getDataAttrAggByInternalId(static_cast<hnswlib::tableint>(signed_node)));
        for (std::size_t attribute = 0; attribute < dimensions; ++attribute) {
            values[attribute] = values[attribute] == 1.0f
                ? 5.0f + maxima[attribute]
                : values[attribute] - means[attribute];
        }
    }
}

struct IndexMemoryStats {
    std::size_t level0_bytes = 0;
    std::size_t upper_level_bytes = 0;
    std::size_t attribute_bytes = 0;
    std::size_t runtime_metadata_bytes = 0;
    std::size_t total() const {
        return level0_bytes + upper_level_bytes + attribute_bytes + runtime_metadata_bytes;
    }
};

inline IndexMemoryStats index_memory_stats(const hnswlib::HierarchicalNSW<float>& index) {
    IndexMemoryStats stats;
    const std::size_t max_elements = index.max_elements_;
    const std::size_t current_elements = index.cur_element_count.load();
    stats.level0_bytes = max_elements * index.size_data_per_element_;
    stats.attribute_bytes = max_elements *
        (index.size_data_attr_per_element_ + index.size_data_attr_agg_per_element_);
    stats.upper_level_bytes = max_elements * sizeof(char*);
    for (std::size_t node = 0; node < current_elements; ++node) {
        if (index.element_levels_[node] > 0) {
            stats.upper_level_bytes +=
                static_cast<std::size_t>(index.element_levels_[node]) * index.size_links_per_element_;
        }
    }
    stats.runtime_metadata_bytes =
        max_elements * sizeof(int) + max_elements * sizeof(std::mutex) +
        hnswlib::HierarchicalNSW<float>::MAX_LABEL_OPERATION_LOCKS * sizeof(std::mutex) +
        max_elements * sizeof(hnswlib::vl_type) +
        current_elements *
            (sizeof(std::pair<const hnswlib::labeltype, hnswlib::tableint>) + sizeof(void*));
    return stats;
}

inline void print_memory_stats(const hnswlib::HierarchicalNSW<float>& index) {
    const auto stats = index_memory_stats(index);
    const std::size_t vector_bytes = index.cur_element_count.load() * index.data_size_;
    std::cout << "index_memory_bytes=" << stats.total()
              << " index_memory_mib=" << std::fixed << std::setprecision(3)
              << static_cast<double>(stats.total()) / (1024.0 * 1024.0)
              << " index_level0_bytes=" << stats.level0_bytes
              << " index_upper_level_bytes=" << stats.upper_level_bytes
              << " index_runtime_metadata_bytes=" << stats.runtime_metadata_bytes
              << " index_attribute_bytes=" << stats.attribute_bytes
              << " vector_storage_bytes=" << vector_bytes << '\n';
}

inline double mean(const std::vector<double>& values) {
    return values.empty() ? 0.0 :
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

}  // namespace rwalks_adapter
