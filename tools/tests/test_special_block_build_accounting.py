from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "UNG" / "codes" / "src"
INCLUDE = ROOT / "UNG" / "codes" / "include"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class SpecialBlockBuildAccountingTests(unittest.TestCase):
    def test_special_block_build_time_fields_are_persisted(self):
        io_cpp = read(SRC / "uni_nav_graph_io.cpp")

        expected_fields = [
            "special_block_metadata_time",
            "special_edge_overlay_time",
            "special_block_indexes_prepare_time",
            "special_edge_intra_build_time",
            "special_edge_inter_build_time",
            "special_blocks_save_time",
        ]

        for field in expected_fields:
            self.assertIn(f'"{field}"', io_cpp)

    def test_special_block_summary_tracks_requested_stage_timers(self):
        header = read(INCLUDE / "ung_special_blocks.h")

        expected_members = [
            "metadata_ms",
            "edge_overlay_ms",
            "block_indexes_prepare_ms",
            "intra_edge_build_ms",
            "inter_edge_build_ms",
            "save_total_ms",
        ]

        for member in expected_members:
            self.assertIn(member, header)

    def test_index_size_includes_special_block_data(self):
        io_cpp = read(SRC / "uni_nav_graph_io.cpp")

        self.assertIn("special_block_size", io_cpp)
        self.assertIn("_special_blocks", io_cpp)
        self.assertIn("_group_id_to_special_block", io_cpp)
        self.assertIn("_point_to_special_block", io_cpp)
        self.assertIn("_special_edges_by_point", io_cpp)
        self.assertIn("_index_size += special_block_size", io_cpp)

    def test_special_inter_edge_optimization_knobs_are_present(self):
        special_cpp = read(SRC / "uni_nav_graph_special_blocks.cpp")
        config = read(ROOT / "experiments" / "cpu_special_blocks" / "config.json")

        expected_envs = [
            "UNG_SPECIAL_INTER_FORCE_GRAPH_PAIR_WORK",
            "UNG_SPECIAL_INTER_SEARCH_EF",
            "UNG_SPECIAL_INTER_SAMPLE_CANDIDATES",
            "UNG_SPECIAL_INTRA_SAMPLE_CANDIDATES",
            "UNG_SPECIAL_INTRA_SAMPLE_MODE",
        ]
        for env in expected_envs:
            self.assertIn(env, special_cpp)

        self.assertIn('"UNG_SPECIAL_INTER_FORCE_GRAPH_PAIR_WORK": "10000"', config)
        self.assertIn('"UNG_SPECIAL_INTER_SAMPLE_CANDIDATES"', config)
        self.assertIn('"UNG_SPECIAL_INTRA_SAMPLE_CANDIDATES"', config)
        self.assertIn('"UNG_SPECIAL_INTRA_SAMPLE_MODE": "hashprune"', config)
        self.assertIn('"UNG_SPECIAL_INTRA_COMPLETE_NX": "50000"', config)
        self.assertIn('"UNG_SPECIAL_INTRA_EXACT_TOPK": "1"', config)
        self.assertIn("force_graph_pair_work", special_cpp)
        self.assertIn("inter_search_ef", special_cpp)
        self.assertIn("sampled_dst_points", special_cpp)
        self.assertIn("build_sampled_vamana_graph_for_points", special_cpp)
        self.assertIn("build_from_candidate_pools", read(ROOT / "UNG" / "codes" / "vamana" / "vamana.h"))
        self.assertIn("inter_insert(id, pruned_list", read(ROOT / "UNG" / "codes" / "vamana" / "vamana.cpp"))
        self.assertIn("sampled_intra_blocks", special_cpp)
        self.assertIn("slowest_pairs", special_cpp)
        self.assertIn("max_pair_work", special_cpp)

    def test_special_intra_profile_log_passes_sample_mode_string(self):
        special_cpp = read(SRC / "uni_nav_graph_special_blocks.cpp")
        fmt_pos = special_cpp.find("intra_sample_mode=%s")
        self.assertNotEqual(fmt_pos, -1)
        args_pos = special_cpp.find("intra_sample_mode.c_str()", fmt_pos)
        next_numeric_arg = special_cpp.find("intra_sample_candidates_env", fmt_pos)
        self.assertNotEqual(args_pos, -1)
        self.assertLess(args_pos, next_numeric_arg)

    def test_special_intra_hashprune_builder_knobs_are_present(self):
        special_cpp = read(SRC / "uni_nav_graph_special_blocks.cpp")
        config = read(ROOT / "experiments" / "cpu_special_blocks" / "config.json")

        expected_envs = [
            "UNG_SPECIAL_HASHPRUNE_LEAF_SIZE",
            "UNG_SPECIAL_HASHPRUNE_REPLICAS",
            "UNG_SPECIAL_HASHPRUNE_RESERVOIR",
            "UNG_SPECIAL_HASHPRUNE_LEAF_TOPK",
            "UNG_SPECIAL_HASHPRUNE_FANOUT",
        ]
        for env in expected_envs:
            self.assertIn(env, special_cpp)

        self.assertIn("HashPruneReservoir", special_cpp)
        self.assertIn("build_hashprune_graph_for_points", special_cpp)
        self.assertIn("hashprune_edges_streamed", special_cpp)
        self.assertIn("hashprune", special_cpp)
        self.assertIn('"UNG_SPECIAL_INTRA_SAMPLE_MODE": "hashprune"', config)
        self.assertIn('"UNG_SPECIAL_HASHPRUNE_LEAF_SIZE"', config)
        self.assertIn('"UNG_SPECIAL_HASHPRUNE_REPLICAS"', config)
        self.assertIn('"UNG_SPECIAL_HASHPRUNE_RESERVOIR"', config)

    def test_hashprune_reservoir_mix_and_degree_diagnostics_are_present(self):
        special_cpp = read(SRC / "uni_nav_graph_special_blocks.cpp")
        config = read(ROOT / "experiments" / "cpu_special_blocks" / "config.json")

        expected_envs = [
            "UNG_SPECIAL_HASHPRUNE_BUCKET_CAP",
            "UNG_SPECIAL_HASHPRUNE_NEAREST",
            "UNG_SPECIAL_HASHPRUNE_MIX_SAMPLE",
            "UNG_SPECIAL_HASHPRUNE_DIAG",
        ]
        for env in expected_envs:
            self.assertIn(env, special_cpp)

        expected_markers = [
            "HashPruneGraphDiagnostics",
            "nearest_capacity",
            "bucket_capacity",
            "hashprune_mix_sample",
            "hashprune_final_avg_degree",
            "hashprune_avg_bucket_coverage",
        ]
        for marker in expected_markers:
            self.assertIn(marker, special_cpp)

        for env in expected_envs:
            self.assertIn(env, config)

    def test_hashprune_navigation_repair_knobs_are_present(self):
        special_cpp = read(SRC / "uni_nav_graph_special_blocks.cpp")
        config = read(ROOT / "experiments" / "cpu_special_blocks" / "config.json")

        expected_envs = [
            "UNG_SPECIAL_HASHPRUNE_NAV_REPAIR",
            "UNG_SPECIAL_HASHPRUNE_REPAIR_2HOP_MIN",
            "UNG_SPECIAL_HASHPRUNE_REPAIR_GRAPH_HOPS",
            "UNG_SPECIAL_HASHPRUNE_REPAIR_HASH_NEAR",
            "UNG_SPECIAL_HASHPRUNE_REPAIR_RESERVOIR_NEAR",
        ]
        for env in expected_envs:
            self.assertIn(env, special_cpp)
            self.assertIn(env, config)

        expected_markers = [
            "hashprune_nav_repair",
            "hashprune_low_2hop_points",
            "hashprune_graph_repair_candidates_added",
            "hashprune_hash_repair_candidates_added",
            "hashprune_reservoir_repair_candidates_added",
            "collect_hash_near_candidates",
            "add_graph_expansion_candidates_to_pool",
        ]
        for marker in expected_markers:
            self.assertIn(marker, special_cpp)

    def test_hashprune_low_degree_repair_knobs_are_present(self):
        special_cpp = read(SRC / "uni_nav_graph_special_blocks.cpp")
        config = read(ROOT / "experiments" / "cpu_special_blocks" / "config.json")

        expected_envs = [
            "UNG_SPECIAL_HASHPRUNE_REPAIR_DEGREE",
            "UNG_SPECIAL_HASHPRUNE_REPAIR_SAMPLE",
            "UNG_SPECIAL_HASHPRUNE_REPAIR_MAX_POINTS",
        ]
        for env in expected_envs:
            self.assertIn(env, special_cpp)
            self.assertIn(env, config)

        expected_markers = [
            "hashprune_repair_degree",
            "hashprune_repaired_points",
            "hashprune_repair_candidates_added",
            "repair_low_final_degree_points",
            "rebuild_hashprune_after_low_degree_repair",
        ]
        for marker in expected_markers:
            self.assertIn(marker, special_cpp)


if __name__ == "__main__":
    unittest.main()
