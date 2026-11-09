#include "common.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

#include <hnswlib/space_l2.h>

int main(int argc, char** argv) {
    try {
        if (argc != 11) {
            std::cerr << "usage: " << argv[0]
                      << " <base.fvecs> <base_labels.txt> <schema.txt> <index.bin>"
                         " <num_threads> <M> <ef_construction> <attr_depth>"
                         " <attr_steps> <random_seed>\n";
            return 2;
        }

        const std::filesystem::path base_path = argv[1];
        const std::filesystem::path base_label_path = argv[2];
        const std::filesystem::path schema_path = argv[3];
        const std::filesystem::path index_path = argv[4];
        const int num_threads = std::stoi(argv[5]);
        const int M = std::stoi(argv[6]);
        const int ef_construction = std::stoi(argv[7]);
        const int attr_depth = std::stoi(argv[8]);
        const int attr_steps = std::stoi(argv[9]);
        const std::uint32_t random_seed = static_cast<std::uint32_t>(std::stoul(argv[10]));
        if (num_threads <= 0 || M <= 0 || ef_construction <= 0 ||
            attr_depth <= 0 || attr_steps <= 0) {
            throw std::runtime_error("threads and RWalks build parameters must be positive");
        }

        rwalks_adapter::MappedFvecs base(base_path);
        if (base.count() == 0) throw std::runtime_error("base vectors must not be empty");
        const auto schema = rwalks_adapter::load_schema(schema_path);
        auto dense_labels = rwalks_adapter::load_dense_labels(
            base_label_path, base.count(), schema, false);

        std::cout << "build params: vectors=" << base.count()
                  << " dimension=" << base.dimension()
                  << " attribute_dimension=" << schema.size()
                  << " num_threads=" << num_threads
                  << " M=" << M
                  << " ef_construction=" << ef_construction
                  << " attr_depth=" << attr_depth
                  << " attr_steps=" << attr_steps
                  << " random_seed=" << random_seed << '\n';

        hnswlib::L2Space space(base.dimension(), schema.size());
        auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &space, base.count(), static_cast<std::size_t>(M),
            static_cast<std::size_t>(ef_construction), random_seed, false);

        const auto graph_begin = std::chrono::steady_clock::now();
        index->addPoint(base.row(0), 0, false, dense_labels.data());
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 64)
        for (std::int64_t row = 1; row < static_cast<std::int64_t>(base.count()); ++row) {
            index->addPoint(
                base.row(static_cast<std::size_t>(row)),
                static_cast<hnswlib::labeltype>(row), false,
                dense_labels.data() + static_cast<std::size_t>(row) * schema.size());
        }
        const auto graph_end = std::chrono::steady_clock::now();
        std::cout << "graph_build_seconds="
                  << std::chrono::duration<double>(graph_end - graph_begin).count() << '\n';

        std::vector<int>().swap(dense_labels);
        const auto walks_begin = std::chrono::steady_clock::now();
        rwalks_adapter::compute_random_walk_attributes(
            *index, attr_depth, attr_steps, 1.0f, num_threads, random_seed);
        const auto walks_end = std::chrono::steady_clock::now();
        std::cout << "random_walk_seconds="
                  << std::chrono::duration<double>(walks_end - walks_begin).count() << '\n';

        std::filesystem::create_directories(index_path.parent_path());
        index->saveIndex(index_path.string());

        std::free(index->data_attr_agg_memory_);
        index->data_attr_agg_memory_ = nullptr;
        index.reset();
        auto loaded_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &space, index_path.string(), false, 0, false);
        std::cout << "loaded_index_elements=" << loaded_index->cur_element_count.load() << '\n';
        rwalks_adapter::print_memory_stats(*loaded_index);
        std::free(loaded_index->data_attr_agg_memory_);
        loaded_index->data_attr_agg_memory_ = nullptr;
        std::cout << "save index in " << index_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RWalks build failed: " << error.what() << '\n';
        return 1;
    }
}
