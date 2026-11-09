#include "common.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>

#include <faiss/IndexPathSeer.h>
#include <faiss/index_io.h>
#include <omp.h>

namespace {

template <typename Vector>
std::size_t vector_capacity_bytes(const Vector& values) {
    return values.capacity() * sizeof(typename Vector::value_type);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 10) {
            std::cerr << "usage: " << argv[0]
                      << " <base.fvecs> <index.bin> <num_threads> <M> <M_exp>"
                         " <ef_construction> <ef_construction_exp> <dist_fn>"
                         " <optimized_building:0|1>\n";
            return 2;
        }

        const std::filesystem::path base_path = argv[1];
        const std::filesystem::path index_path = argv[2];
        const int num_threads = std::stoi(argv[3]);
        const int M = std::stoi(argv[4]);
        const int M_exp = std::stoi(argv[5]);
        const int ef_construction = std::stoi(argv[6]);
        const int ef_construction_exp = std::stoi(argv[7]);
        const std::string dist_fn = argv[8];
        const bool optimized_building = std::stoi(argv[9]) != 0;
        if (num_threads <= 0 || M <= 0 || M_exp <= 0 || ef_construction <= 0 ||
            ef_construction_exp <= 0) {
            throw std::runtime_error("thread and PathSeer build parameters must be positive");
        }

        project_adapter::FvecsData base = project_adapter::load_fvecs(base_path);
        if (project_adapter::uses_cosine_normalization(dist_fn)) {
            project_adapter::normalize_rows(base);
        }
        omp_set_num_threads(num_threads);

        std::cout << "build params: vectors=" << base.count
                  << " dimension=" << base.dimension
                  << " num_threads=" << num_threads
                  << " M=" << M
                  << " M_exp=" << M_exp
                  << " ef_construction=" << ef_construction
                  << " ef_construction_exp=" << ef_construction_exp
                  << " dist_fn=" << dist_fn
                  << " optimized_building=" << optimized_building << '\n';

        auto index = std::make_unique<faiss::IndexPathSeerFlat>(
            base.dimension, M, M_exp, project_adapter::parse_metric(dist_fn));
        index->pathseer.efConstruction = ef_construction;
        index->pathseer.efConstructionExp = ef_construction_exp;
        index->set_using_optimized_building(optimized_building);
        index->add(static_cast<faiss::idx_t>(base.count), base.values.data());

        std::filesystem::create_directories(index_path.parent_path());
        faiss::write_index(index.get(), index_path.c_str());

        const auto& graph = index->pathseer;
        std::size_t level0_slots = 0;
        std::size_t upper_level_slots = 0;
        const std::size_t level0_capacity = static_cast<std::size_t>(graph.nb_neighbors(0));
        for (std::size_t node = 0; node + 1 < graph.offsets.size(); ++node) {
            const std::size_t node_slots = graph.offsets[node + 1] - graph.offsets[node];
            const std::size_t node_level0_slots = std::min(node_slots, level0_capacity);
            level0_slots += node_level0_slots;
            upper_level_slots += node_slots - node_level0_slots;
        }
        const std::size_t level0_bytes =
            level0_slots * sizeof(faiss::PathSeer::NeighNode);
        const std::size_t upper_level_bytes =
            upper_level_slots * sizeof(faiss::PathSeer::NeighNode);
        const std::size_t runtime_metadata_bytes =
            vector_capacity_bytes(graph.levels) + vector_capacity_bytes(graph.offsets) +
            vector_capacity_bytes(graph.offset_neighbor) +
            vector_capacity_bytes(graph.neighbor_boundaries) +
            vector_capacity_bytes(graph.cum_nneighbor_per_level) +
            vector_capacity_bytes(graph.assign_probas);
        const std::size_t vector_bytes =
            base.count * static_cast<std::size_t>(base.dimension) * sizeof(float);
        const std::size_t index_memory_bytes =
            vector_bytes + level0_bytes + upper_level_bytes + runtime_metadata_bytes;

        std::cout << "index_memory_bytes=" << index_memory_bytes
                  << " index_memory_mib=" << std::fixed << std::setprecision(3)
                  << static_cast<double>(index_memory_bytes) / (1024.0 * 1024.0)
                  << " index_level0_bytes=" << level0_bytes
                  << " index_upper_level_bytes=" << upper_level_bytes
                  << " index_runtime_metadata_bytes=" << runtime_metadata_bytes
                  << " vector_storage_bytes=" << vector_bytes << '\n';
        std::cout << "save index in " << index_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PathSeer build failed: " << error.what() << '\n';
        return 1;
    }
}
