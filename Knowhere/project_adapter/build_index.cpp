#include "common.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>

#include <omp.h>

namespace {

template <typename Vector>
std::size_t capacity_bytes(const Vector& values) {
    return values.capacity() * sizeof(typename Vector::value_type);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 10) {
            std::cerr << "usage: " << argv[0]
                      << " <base.fvecs> <base_labels.txt> <index.bin> <filters.bin>"
                         " <num_threads> <M> <ef_construction> <dist_fn> <add_batch_size>\n";
            return 2;
        }
        const std::filesystem::path base_path = argv[1];
        const std::filesystem::path labels_path = argv[2];
        const std::filesystem::path index_path = argv[3];
        const std::filesystem::path filters_path = argv[4];
        const int num_threads = std::stoi(argv[5]);
        const int M = std::stoi(argv[6]);
        const int ef_construction = std::stoi(argv[7]);
        const std::string dist_fn = argv[8];
        std::size_t add_batch_size = std::stoull(argv[9]);
        if (num_threads <= 0 || M < 2 || ef_construction <= 0)
            throw std::runtime_error("threads, M and ef_construction must be positive; add_batch_size may be zero");

        auto base = milvus_adapter::load_fvecs(base_path);
        if (add_batch_size == 0) add_batch_size = std::max<std::size_t>(base.count, 1);
        if (milvus_adapter::uses_cosine_normalization(dist_fn)) milvus_adapter::normalize_rows(base);
        auto label_rows = milvus_adapter::load_label_rows(labels_path);
        if (label_rows.size() != base.count)
            throw std::runtime_error("base-label row count does not match base vectors");
        omp_set_num_threads(num_threads);

        std::cout << "build params: vectors=" << base.count << " dimension=" << base.dimension
                  << " num_threads=" << num_threads << " M=" << M
                  << " ef_construction=" << ef_construction << " dist_fn=" << dist_fn
                  << " add_batch_size=" << add_batch_size << '\n';

        auto index = std::make_unique<milvus_adapter::kh::IndexHNSWFlat>(
            base.dimension, M, milvus_adapter::parse_metric(dist_fn));
        index->hnsw.efConstruction = ef_construction;
        for (std::size_t begin = 0; begin < base.count; begin += add_batch_size) {
            const std::size_t count = std::min(add_batch_size, base.count - begin);
            index->add(static_cast<faiss::idx_t>(count), base.values.data() + begin * base.dimension);
        }
        auto filters = milvus_adapter::build_filter_index(label_rows);
        milvus_adapter::save_index(*index, index_path);
        milvus_adapter::save_filter_index(filters, filters_path);

        index.reset();
        base.values.clear();
        base.values.shrink_to_fit();
        label_rows.clear();
        label_rows.shrink_to_fit();
        filters = milvus_adapter::FilterIndex{};

        auto loaded_index = milvus_adapter::load_index(index_path);
        const auto loaded_filters = milvus_adapter::load_filter_index(filters_path);
        const auto& graph = loaded_index->hnsw;
        std::size_t level0_slots = 0, upper_slots = 0;
        const std::size_t level0_capacity = static_cast<std::size_t>(graph.nb_neighbors(0));
        for (std::size_t node = 0; node + 1 < graph.offsets.size(); ++node) {
            const std::size_t slots = graph.offsets[node + 1] - graph.offsets[node];
            level0_slots += std::min(slots, level0_capacity);
            upper_slots += slots - std::min(slots, level0_capacity);
        }
        const std::size_t level0_bytes = level0_slots * sizeof(milvus_adapter::kh::HNSW::storage_idx_t);
        const std::size_t upper_bytes = upper_slots * sizeof(milvus_adapter::kh::HNSW::storage_idx_t);
        const std::size_t metadata_bytes = capacity_bytes(graph.levels) + capacity_bytes(graph.offsets) +
            capacity_bytes(graph.cum_nneighbor_per_level) + capacity_bytes(graph.assign_probas);
        const std::size_t vector_bytes = static_cast<std::size_t>(loaded_index->ntotal) * static_cast<std::size_t>(loaded_index->d) * sizeof(float);
        const std::size_t filter_bytes = milvus_adapter::filter_memory_bytes(loaded_filters);
        const std::size_t total_bytes = vector_bytes + level0_bytes + upper_bytes + metadata_bytes + filter_bytes;
        std::cout << "index_memory_bytes=" << total_bytes
                  << " index_memory_mib=" << std::fixed << std::setprecision(3)
                  << static_cast<double>(total_bytes) / (1024.0 * 1024.0)
                  << " index_level0_bytes=" << level0_bytes
                  << " index_upper_level_bytes=" << upper_bytes
                  << " index_runtime_metadata_bytes=" << metadata_bytes
                  << " index_filter_bytes=" << filter_bytes
                  << " vector_storage_bytes=" << vector_bytes << '\n';
        std::cout << "saved Milvus HNSW index in " << index_path << " and filters in " << filters_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Milvus HNSW build failed: " << error.what() << '\n';
        return 1;
    }
}
