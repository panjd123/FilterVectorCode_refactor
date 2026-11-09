// Read-only accounting probe; compiled with the existing UNG search target's ABI.
// Private access is confined to this translation unit and changes no class layout.
#include <bits/stdc++.h>
#include <boost/dynamic_bitset.hpp>
#include <roaring/roaring.hh>
#include <roaring/containers/containers.h>
#define private public
#include "uni_nav_graph.h"
#undef private

using namespace ANNS;
using Bytes = uint64_t;

template<class T> Bytes flat(const std::vector<T>& v) {
    return v.capacity() * sizeof(T);
}
template<class T> Bytes nested(const std::vector<std::vector<T>>& v) {
    Bytes b = flat(v);
    for (const auto& row : v) b += flat(row);
    return b;
}
template<class Map> Bytes hash_storage(const Map& m) {
    // Integral keys use libstdc++'s non-cached hash nodes in this target.
    using Node = std::__detail::_Hash_node<typename Map::value_type, false>;
    return (m.bucket_count() > 1 ? m.bucket_count() * sizeof(void*) : 0)
        + m.size() * sizeof(Node);
}
Bytes roaring_dynamic(const roaring::Roaring& r) {
    using namespace roaring::internal;
    const auto& a = r.roaring.high_low_container;
    Bytes b = Bytes(a.allocation_size) * (sizeof(void*) + sizeof(uint16_t) + sizeof(uint8_t));
    for (int i = 0; i < a.size; ++i) {
        const auto* c = a.containers[i];
        auto type = a.typecodes[i];
        if (type == SHARED_CONTAINER_TYPE)
            throw std::runtime_error("Unexpected shared CRoaring container: deduplication required");
        switch (type) {
        case ARRAY_CONTAINER_TYPE: {
            const auto* p = reinterpret_cast<const array_container_t*>(c);
            b += sizeof(*p) + Bytes(p->capacity) * sizeof(uint16_t); break;
        }
        case BITSET_CONTAINER_TYPE:
            b += sizeof(bitset_container_t) + 8192; break;
        case RUN_CONTAINER_TYPE: {
            const auto* p = reinterpret_cast<const run_container_t*>(c);
            b += sizeof(*p) + Bytes(p->capacity) * sizeof(rle16_t); break;
        }
        default: throw std::runtime_error("Unknown CRoaring container type");
        }
    }
    return b;
}
Bytes roaring_vector(const std::vector<roaring::Roaring>& v) {
    Bytes b = flat(v);
    for (const auto& r : v) b += roaring_dynamic(r);
    return b;
}
Bytes trie_dynamic(const TrieIndex& t) {
    Bytes b = nested(t._label_to_nodes);
    const auto count_node = [&](const TrieNode& n) {
        // Trie loader uses make_shared: node and control block share one allocation.
        using Allocation = std::_Sp_counted_ptr_inplace<TrieNode, std::allocator<void>, __gnu_cxx::_S_atomic>;
        return sizeof(Allocation) + hash_storage(n.children);
    };
    // The loader places every node, including the root, once in the postings.
    for (const auto& row : t._label_to_nodes)
        for (const auto& p : row) b += count_node(*p);
    return b;
}
Bytes group_trie_dynamic(const SpecialBlockTrieIndex& t) {
    Bytes b = flat(t.nodes_) + flat(t.child_ids_) + flat(t.subtree_has_block_)
        + flat(t.group_terminal_nodes_) + flat(t.terminal_ancestor_offsets_)
        + flat(t.terminal_ancestor_groups_) + flat(t.root_entry_groups_)
        + hash_storage(t.label_postings_) + hash_storage(t.label_group_bitsets_);
    for (const auto& p : t.label_postings_) b += flat(p.second);
    for (const auto& p : t.label_group_bitsets_) b += roaring_dynamic(p.second);
    return b;
}
Bytes graph_bytes(const Graph& g, bool locks) {
    Bytes b = sizeof(g) + Bytes(g._num_points) * sizeof(NeighborList);
    if (locks) b += Bytes(g._num_points) * sizeof(std::mutex);
    for (IdxType i = 0; i < g._num_points; ++i)
        if (g.neighbors[i].data() != g.neighbors[i].inline_data_)
            b += g.neighbors[i].capacity() * sizeof(IdxType);
    return b;
}
Bytes blocks_dynamic(const std::vector<SpecialBlock>& v) {
    Bytes b = flat(v);
    for (const auto& block : v)
        b += flat(block.root_labels) + flat(block.common_labels)
            + flat(block.member_group_ids) + flat(block.child_block_ids);
    return b;
}
template<class T> void release(T& value) { T{}.swap(value); }

void prune(UniNavGraph& g, bool ptnc) {
    // Remove data used only by building, alternate algorithms or diagnostics.
    g._global_graph.reset();
    delete[] g._graph->neighbor_locks;
    g._graph->neighbor_locks = nullptr;
    auto* storage = dynamic_cast<Storage<float>*>(g._base_storage.get());
    delete[] storage->label_sets;
    storage->label_sets = nullptr;
    release(g._group_id_to_vec_ids);
    release(g._old_to_new_vec_ids);
    g._label_nav_graph.reset();
    g._group_topology_graph.reset();
    release(g._lng_descendants_rb);
    release(g._covered_sets_rb);
    g._trie_static_metrics.label_frequency.clear();
    g._special_block_trie_index.clear();
    release(g._special_block_router_centroids);
    release(g._special_block_router_radii);
    release(g._special_heavy_edges_by_point);
    release(g._special_heavy_edges_csr.offsets);
    release(g._special_heavy_edges_csr.edges);
    if (ptnc) {
        release(g._trie_index._label_to_nodes);
        g._trie_index._root.reset();
    } else {
        g._group_trie_index.clear();
        release(g._new_vec_id_to_group_id);
    }
}

// Synthetic queries use real indexed vectors and their labels. Both a full
// predicate and a one-label predicate are checked across the index ID range.
std::shared_ptr<IStorage> make_queries(UniNavGraph& g) {
    auto q = create_storage("float", false);
    auto* s = dynamic_cast<Storage<float>*>(q.get());
    s->num_points = 12;
    s->dim = g._base_storage->get_dim();
    s->prefetch_byte_num = s->dim * sizeof(float);
    const Bytes bytes = Bytes(s->num_points) * s->dim * sizeof(float);
    s->vecs = static_cast<float*>(std::aligned_alloc(64, (bytes + 63) / 64 * 64));
    if (!s->vecs) throw std::bad_alloc();
    s->label_sets = new std::vector<LabelType>[s->num_points];
    for (IdxType i = 0; i < s->num_points; ++i) {
        const IdxType id = Bytes(i / 2) * (g._num_points - 1) / 5;
        std::memcpy(s->vecs + Bytes(i) * s->dim, g._base_storage->get_vector(id), s->dim * sizeof(float));
        s->label_sets[i] = g._base_storage->get_label_set(id);
        if (i % 2 && !s->label_sets[i].empty())
            s->label_sets[i] = {s->label_sets[i].back()};
    }
    return q;
}
std::vector<std::pair<IdxType,float>> run_queries(UniNavGraph& g,
    std::shared_ptr<IStorage>& queries, bool ptnc, bool disable_stats) {
    setenv("UNG_DISABLE_ENTRY_ROUTE_STATS", disable_stats ? "1" : "0", 1);
    SearchRuntimeConfig runtime;
    runtime.num_threads = 1; runtime.Lsearch = 64; runtime.K = 10;
    runtime.num_entry_points = 1; runtime.scenario = "containment";
    runtime.force_use_alg = 1;
    runtime.entry_group_strategy = ptnc ? EntryGroupStrategy::Trie : EntryGroupStrategy::OptimizedLng;
    runtime.special_block_search = ptnc; runtime.special_light_stats = true;
    std::shared_ptr<DistanceHandler> distance = get_distance_handler("float", "L2");
    std::vector<std::pair<IdxType,float>> results(queries->get_num_points() * runtime.K, {UINT32_MAX, 0});
    std::vector<float> cmps(queries->get_num_points());
    std::vector<QueryStats> stats;
    std::srand(0);
    g.search_hybrid(queries, distance, runtime, results.data(), cmps, stats);
    return results;
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string folder = argv[1], dataset = argv[2];
    const bool ptnc = std::string(argv[3]) == "PTNC";
    UniNavGraph g;
    g.load(folder + "/", "", "float", "", "", dataset);
    if (ptnc) g.load_special_block_index(folder + "/../block_index_files/");
    auto queries = make_queries(g);
    // Performance runs already disable route statistics. Some UNG indexes were
    // built without coverage sidecars, so enabling statistics would throw.
    const auto before = run_queries(g, queries, ptnc, true);
    std::map<std::string,Bytes> included, excluded;
    included["index_object"] = sizeof(g);
    included["vector_storage_object"] = sizeof(Storage<float>);
    included["vectors"] = Bytes(g._num_points) * g._base_storage->get_dim() * sizeof(float);
    included["graph"] = graph_bytes(*g._graph, false);
    included["group_labels"] = nested(g._group_id_to_label_set);
    included["group_ranges"] = flat(g._group_id_to_range);
    included["group_entry_points"] = flat(g._group_entry_points);
    included["new_to_old_ids"] = flat(g._new_to_old_vec_ids);
    included["distance_handler"] = sizeof(FloatL2DistanceHandler);
    included["shared_control_blocks"] = 2 * sizeof(std::_Sp_counted_base<__gnu_cxx::_S_atomic>)
        + sizeof(std::_Sp_counted_ptr<DistanceHandler*, __gnu_cxx::_S_atomic>);
    if (ptnc) {
        included["entry_group_trie"] = group_trie_dynamic(g._group_trie_index);
        included["point_group_ids"] = flat(g._new_vec_id_to_group_id);
        included["blocks"] = blocks_dynamic(g._special_blocks);
        included["group_block_maps"] = nested(g._group_to_special_block_by_level)
            + flat(g._group_is_special_block_root) + flat(g._group_is_trivial_special_block_root);
        included["point_block_maps"] = nested(g._point_to_special_block_by_level) + flat(g._point_is_special_block_root);
        included["special_edges"] = nested(g._special_edges_by_point) + flat(g._special_edges_csr.offsets) + flat(g._special_edges_csr.edges);
        excluded["legacy_trie_alternate_entry_strategy"] = trie_dynamic(g._trie_index);
    } else {
        included["entry_legacy_trie"] = trie_dynamic(g._trie_index);
        excluded["group_trie_alternate_entry_strategy"] = group_trie_dynamic(g._group_trie_index);
        excluded["point_group_ids_unused"] = flat(g._new_vec_id_to_group_id);
    }
    if (g._global_graph) excluded["legacy_global_graph"] = graph_bytes(*g._global_graph, true);
    excluded["graph_build_locks"] = Bytes(g._num_points) * sizeof(std::mutex);
    Bytes point_labels = Bytes(g._num_points) * sizeof(std::vector<LabelType>);
    for (IdxType i = 0; i < g._num_points; ++i) point_labels += flat(g._base_storage->get_label_set(i));
    excluded["per_point_labels_redundant_with_group_labels"] = point_labels;
    excluded["group_original_point_ids_build_only"] = nested(g._group_id_to_vec_ids);
    excluded["reverse_id_mapping_other_search_modes"] = flat(g._old_to_new_vec_ids);
    if (g._label_nav_graph) {
        const auto& l = *g._label_nav_graph;
        excluded["label_graph_build_and_statistics"] = sizeof(l) + nested(l.in_neighbors)
            + nested(l.out_neighbors) + nested(l.covered_sets) + nested(l._lng_descendants)
            + flat(l.coverage_ratio) + flat(l.in_degree) + flat(l.out_degree) + flat(l._lng_descendants_num);
    }
    excluded["descendant_roaring_route_statistics"] = roaring_vector(g._lng_descendants_rb);
    excluded["covered_point_roaring_route_statistics"] = roaring_vector(g._covered_sets_rb);
    excluded["block_trie_build_only"] = group_trie_dynamic(g._special_block_trie_index);
    excluded["optional_block_router_bounds"] = flat(g._special_block_router_centroids) + flat(g._special_block_router_radii);
    excluded["optional_heavy_edges"] = nested(g._special_heavy_edges_by_point) + flat(g._special_heavy_edges_csr.offsets) + flat(g._special_heavy_edges_csr.edges);
    prune(g, ptnc);
    const auto after = run_queries(g, queries, ptnc, true);
    if (before != after) throw std::runtime_error("Search results changed after pruning");
    Bytes total = 0;
    for (const auto& p : included) {
        total += p.second;
        std::cout << "MEMORY_COMPONENT included " << p.first << " " << p.second << "\n";
    }
    for (const auto& p : excluded)
        std::cout << "MEMORY_COMPONENT excluded " << p.first << " " << p.second << "\n";
    std::cout << "SEARCH_MEMORY bytes=" << total << " vector_bytes=" << included["vectors"]
        << " queries_verified=12 result_pairs_verified=" << before.size()
        << " neighbor_list_sizeof=" << sizeof(NeighborList) << std::endl;
    std::_Exit(0);
}
