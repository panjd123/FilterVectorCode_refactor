#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/MetricType.h>

namespace project_adapter {

struct FvecsData {
    std::size_t count = 0;
    int dimension = 0;
    std::vector<float> values;
};

inline FvecsData load_fvecs(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open fvecs file: " + path.string());
    }
    const std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (file_size < static_cast<std::streamsize>(sizeof(std::int32_t))) {
        throw std::runtime_error("fvecs file is empty or truncated: " + path.string());
    }

    std::int32_t dimension = 0;
    input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    if (!input || dimension <= 0) {
        throw std::runtime_error("invalid fvecs dimension in: " + path.string());
    }
    const std::uint64_t record_bytes = sizeof(std::int32_t) +
        static_cast<std::uint64_t>(dimension) * sizeof(float);
    if (static_cast<std::uint64_t>(file_size) % record_bytes != 0) {
        throw std::runtime_error("fvecs file size is not a whole number of records: " + path.string());
    }

    FvecsData result;
    result.count = static_cast<std::size_t>(static_cast<std::uint64_t>(file_size) / record_bytes);
    result.dimension = dimension;
    result.values.resize(result.count * static_cast<std::size_t>(dimension));

    input.seekg(0, std::ios::beg);
    for (std::size_t row = 0; row < result.count; ++row) {
        std::int32_t row_dimension = 0;
        input.read(reinterpret_cast<char*>(&row_dimension), sizeof(row_dimension));
        if (!input || row_dimension != dimension) {
            throw std::runtime_error("inconsistent fvecs dimension at row " + std::to_string(row));
        }
        input.read(
            reinterpret_cast<char*>(result.values.data() + row * dimension),
            static_cast<std::streamsize>(dimension) * sizeof(float));
        if (!input) {
            throw std::runtime_error("truncated fvecs payload at row " + std::to_string(row));
        }
    }
    return result;
}

inline std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline faiss::MetricType parse_metric(const std::string& metric_name) {
    const std::string metric = lower_copy(metric_name);
    if (metric == "l2" || metric == "euclidean") {
        return faiss::METRIC_L2;
    }
    if (metric == "ip" || metric == "inner_product" || metric == "inner-product" ||
        metric == "cosine") {
        return faiss::METRIC_INNER_PRODUCT;
    }
    throw std::runtime_error("unsupported distance function: " + metric_name);
}

inline bool uses_cosine_normalization(const std::string& metric_name) {
    return lower_copy(metric_name) == "cosine";
}

inline void normalize_rows(FvecsData& data) {
    for (std::size_t row = 0; row < data.count; ++row) {
        float* values = data.values.data() + row * data.dimension;
        double squared_norm = 0.0;
        for (int column = 0; column < data.dimension; ++column) {
            squared_norm += static_cast<double>(values[column]) * values[column];
        }
        if (squared_norm == 0.0) {
            continue;
        }
        const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
        for (int column = 0; column < data.dimension; ++column) {
            values[column] *= inverse_norm;
        }
    }
}

inline std::vector<int> parse_label_line(std::string line) {
    for (char& c : line) {
        if (c == ',') {
            c = ' ';
        }
    }
    std::vector<int> labels;
    std::size_t position = 0;
    while (position < line.size()) {
        while (position < line.size() && std::isspace(static_cast<unsigned char>(line[position]))) {
            ++position;
        }
        if (position == line.size()) {
            break;
        }
        std::size_t consumed = 0;
        const int label = std::stoi(line.substr(position), &consumed);
        labels.push_back(label);
        position += consumed;
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    return labels;
}

inline std::vector<std::vector<int>> load_label_rows(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open label file: " + path.string());
    }
    std::vector<std::vector<int>> rows;
    std::string line;
    while (std::getline(input, line)) {
        rows.push_back(parse_label_line(line));
    }
    return rows;
}

inline bool contains_all(const std::vector<int>& base, const std::vector<int>& query) {
    return std::includes(base.begin(), base.end(), query.begin(), query.end());
}

}  // namespace project_adapter
