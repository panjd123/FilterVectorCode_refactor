#!/usr/bin/env python3

from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_search_comparison


class BlockThresholdAblationConfigTest(unittest.TestCase):
    def test_saved_configs_use_one_query_task_and_threshold_indexes(self):
        config_names = {
            "Amazon": "config_amazon_block_num.json",
            "Music": "config_music_block_num.json",
            "VImg": "config_VImg_block_num.json",
        }
        expected_indexes = [
            "Trie_block_100",
            "Trie_block_500",
            "Trie_block_1000",
            "Trie_block_5000",
            "Trie_block_10000",
        ]

        for dataset, config_name in config_names.items():
            with self.subTest(dataset=dataset):
                config_path = Path(__file__).resolve().parent / config_name
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["datasets"],
                    [{"dataset": dataset,
                      "query_task": "query_minlen1_sel10to40pct"}],
                )
                self.assertEqual(
                    [method["index_name"] for method in config["methods"]],
                    expected_indexes,
                )


class NativeUngCampaignConfigTest(unittest.TestCase):
    def test_task_specific_lsearch_ranges(self):
        config_names = {
            "Amazon": "config_amazon_ung.json",
            "BReviews": "config_breviews_ung.json",
            "Laion": "config_laion_ung.json",
            "Music": "config_music_ung.json",
            "Tiktok": "config_tiktok_ung.json",
            "VImg": "config_VImg_ung.json",
            "Genome": "config_genome_ung.json",
            "Reviews": "config_reviews_ung.json",
        }
        expected = {
            "query_minlen1_sel0to10pct": (1000, 20000, 1000, 20),
            "query_minlen1_sel10to40pct": (10000, 100000, 10000, 10),
            "query_minlen1_sel40to100pct": (10000, 100000, 10000, 10),
        }

        for dataset, config_name in config_names.items():
            with self.subTest(dataset=dataset):
                config_path = Path(__file__).resolve().parent / config_name
                config = json.loads(config_path.read_text(encoding="utf-8"))
                for dataset_cfg in config["datasets"]:
                    task = dataset_cfg["query_task"]
                    search = run_search_comparison.effective_search_cfg(
                        config, dataset_cfg
                    )
                    start, end, step, count = expected[task]
                    self.assertEqual(
                        (search["lsearch_start"], search["lsearch_end"],
                         search["lsearch_step"]),
                        (start, end, step),
                    )
                    values = run_search_comparison.lsearch_values(search)
                    self.assertEqual((values[0], values[-1], len(values)),
                                     (start, end, count))


class BlockDegreeAblationConfigTest(unittest.TestCase):
    def test_configs_use_one_query_task_and_three_degree_indexes(self):
        config_names = {
            "Amazon": "config_amazon_block_M.json",
            "Music": "config_music_block_M.json",
            "VImg": "config_VImg_block_M.json",
        }
        expected_indexes = [
            "Trie_block_500_M_16",
            "Trie_block_500_M_32",
            "Trie_block_500_M_64",
        ]

        for dataset, config_name in config_names.items():
            with self.subTest(dataset=dataset):
                config_path = Path(__file__).resolve().parent / config_name
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["datasets"],
                    [{"dataset": dataset,
                      "query_task": "query_minlen1_sel10to40pct"}],
                )
                self.assertEqual(
                    [method["index_name"] for method in config["methods"]],
                    expected_indexes,
                )


class BlockCrossEdgeAblationConfigTest(unittest.TestCase):
    def test_configs_use_one_query_task_and_three_cross_edge_indexes(self):
        config_names = {
            "Amazon": "config_amazon_block_cross_block.json",
            "Music": "config_music_block_cross_block.json",
            "VImg": "config_VImg_block_cross_block.json",
        }
        expected_indexes = [
            "Trie_block_500_M_32_cross_block_2",
            "Trie_block_500_M_32_cross_block_4",
            "Trie_block_500_M_32_cross_block_6",
        ]

        for dataset, config_name in config_names.items():
            with self.subTest(dataset=dataset):
                config_path = Path(__file__).resolve().parent / config_name
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["datasets"],
                    [{"dataset": dataset,
                      "query_task": "query_minlen1_sel10to40pct"}],
                )
                self.assertEqual(
                    [method["index_name"] for method in config["methods"]],
                    expected_indexes,
                )


class ResultQueryDirNameTest(unittest.TestCase):
    def test_appends_lsearch_start_step_end_to_query_task(self):
        self.assertEqual(
            run_search_comparison.result_query_dir_name(
                "query_minlen1_cov10k",
                {
                    "lsearch_start": 100,
                    "lsearch_step": 100,
                    "lsearch_end": 1000,
                },
            ),
            "query_minlen1_cov10k_100_100_1000",
        )

    def test_uses_prefix_metadata_when_prefix_precedes_main_sweep(self):
        self.assertEqual(
            run_search_comparison.result_query_dir_name(
                "query_minlen1_cov10k",
                {
                    "lsearch_start": 1000,
                    "lsearch_step": 1000,
                    "lsearch_end": 20000,
                    "lsearch_prefix_start": 100,
                    "lsearch_prefix_end": 1000,
                    "lsearch_prefix_step": 100,
                },
            ),
            "query_minlen1_cov10k_100_100_20000",
        )


class LsearchValuesTest(unittest.TestCase):
    def test_prepends_dense_low_budget_range_without_duplicate_boundary(self):
        self.assertEqual(
            run_search_comparison.lsearch_values(
                {
                    "lsearch_start": 1000,
                    "lsearch_step": 1000,
                    "lsearch_end": 3000,
                    "lsearch_prefix_start": 100,
                    "lsearch_prefix_end": 1000,
                    "lsearch_prefix_step": 100,
                }
            ),
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000],
        )

    def test_method_range_below_prefix_start_is_not_changed(self):
        self.assertEqual(
            run_search_comparison.lsearch_values(
                {
                    "lsearch_start": 10,
                    "lsearch_step": 10,
                    "lsearch_end": 100,
                    "lsearch_prefix_start": 100,
                    "lsearch_prefix_end": 1000,
                    "lsearch_prefix_step": 100,
                }
            ),
            list(range(10, 101, 10)),
        )


class RunCommandLoggingTest(unittest.TestCase):
    def test_run_command_can_stream_without_writing_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "command.log"

            run_search_comparison.run_command(
                [sys.executable, "-c", "print('child output')"],
                None,
            )

            self.assertFalse(log_path.exists())


class NativeUngIndexTest(unittest.TestCase):
    @staticmethod
    def write_native_index(index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        for name in run_search_comparison.NATIVE_UNG_INDEX_FILES:
            (index_dir / name).write_text("")
        (index_dir / "meta").write_text(
            "base_group_topology=lng\n"
            "hierarchy_layers=\n"
            "lng_constructed=1\n"
        )

    def test_complete_lng_index_skips_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "results" / "Tiny" / "index" / "UNG" / "index_files"
            self.write_native_index(index_dir)
            cfg = {
                "auto_build_ung": True,
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(root / "build"),
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {
                "name": "UNG",
                "index_name": "UNG",
                "entry_group_provider": "cpu_min_super_sets",
                "special_block_search": False,
            }

            with mock.patch.object(run_search_comparison, "run_command") as run:
                run_search_comparison.ensure_native_ung_index(
                    cfg, dataset_cfg, method
                )

            run.assert_not_called()

    def test_missing_index_builds_native_lng_without_special_block_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_bin = root / "build" / "apps" / "build_UNG_index"
            build_bin.parent.mkdir(parents=True)
            build_bin.write_text("")
            data_dir = root / "data" / "Tiny"
            data_dir.mkdir(parents=True)
            (data_dir / "Tiny_base.bin").write_text("")
            (data_dir / "Tiny_base_labels.txt").write_text("")
            index_dir = root / "results" / "Tiny" / "index" / "UNG" / "index_files"
            captured = {}

            def fake_run_command(cmd, output_file, env=None):
                captured["cmd"] = cmd
                captured["output_file"] = output_file
                captured["env"] = env
                self.write_native_index(index_dir)

            cfg = {
                "auto_build_ung": True,
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(root / "build"),
                "build": {
                    "num_threads": 7,
                    "max_degree": 32,
                    "Lbuild": 100,
                    "alpha": 1.2,
                    "num_cross_edges": 4,
                },
                "ung_env": {"UNG_BASE_GROUP_TOPOLOGY": "lng"},
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {
                "name": "UNG",
                "index_name": "UNG",
                "entry_group_provider": "cpu_min_super_sets",
                "special_block_search": False,
            }

            contaminated_env = {
                "UNG_SPECIAL_BLOCKS": "1",
                "UNG_SPECIAL_BLOCK_PARTITION": "trie",
                "UNG_HIERARCHY_LAYERS": "100:trie",
            }
            with mock.patch.dict(
                run_search_comparison.os.environ, contaminated_env, clear=False
            ), mock.patch.object(
                run_search_comparison, "run_command", fake_run_command
            ):
                run_search_comparison.ensure_native_ung_index(
                    cfg, dataset_cfg, method
                )

            self.assertEqual(captured["cmd"][0], str(build_bin))
            self.assertIn("--index_path_prefix", captured["cmd"])
            self.assertEqual(
                captured["env"]["UNG_BASE_GROUP_TOPOLOGY"], "lng"
            )
            self.assertNotIn("UNG_SKIP_COVERAGE", captured["env"])
            self.assertNotIn("UNG_HIERARCHY_LAYERS", captured["env"])
            self.assertFalse(
                any(key.startswith("UNG_SPECIAL_") for key in captured["env"])
            )
            self.assertEqual(
                captured["output_file"],
                root / "results" / "Tiny" / "index" / "UNG" / "others" / "ung_build.log",
            )

    def test_native_search_never_loads_adjacent_block_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "results" / "Tiny" / "index" / "UNG" / "index_files"
            block_dir = index_dir.parent / "block_index_files"
            query_dir = root / "data" / "Tiny" / "query_task"
            index_dir.mkdir(parents=True)
            block_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            (index_dir / "meta").write_text("base_group_topology=lng\n")
            (block_dir / "meta").write_text("base_group_topology=trie\n")
            query_bin = query_dir / "Tiny_query.bin"
            gt_file = query_dir / "gt.bin"
            query_bin.write_text("")
            gt_file.write_text("")
            captured = {}

            def fake_run_command(cmd, output_file, env=None):
                captured["cmd"] = cmd
                captured["env"] = env
                raw_dir = (
                    root
                    / "results"
                    / "Tiny"
                    / "results"
                    / "UNG"
                    / "query_task_10_10_10"
                    / "results"
                )
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "search_time_summary.csv").write_text(
                    "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n"
                    "10,0,1,1\n"
                )

            cfg = {
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(root / "build"),
                "search": {
                    "lsearch_start": 10,
                    "lsearch_step": 10,
                    "lsearch_end": 10,
                    "num_threads": 1,
                    "K": 1,
                    "num_repeats": 1,
                    "num_entry_points": 1,
                    "efs_start": 1,
                    "efs_step_slow": 1,
                    "efs_step_fast": 1,
                    "lsearch_threshold": 10,
                },
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {
                "name": "UNG",
                "index_name": "UNG",
                "entry_group_provider": "cpu_min_super_sets",
                "special_block_search": False,
            }

            with mock.patch.dict(
                run_search_comparison.os.environ,
                {"UNG_SPECIAL_BLOCK_SEARCH": "1"},
                clear=False,
            ), mock.patch.object(
                run_search_comparison, "run_command", fake_run_command
            ):
                run_search_comparison.run_search(
                    cfg, dataset_cfg, method, query_bin, gt_file, 1
                )

            self.assertNotIn("--block_index_path_prefix", captured["cmd"])
            self.assertNotIn("UNG_SPECIAL_BLOCK_SEARCH", captured["env"])


class RunSearchMethodConfigTest(unittest.TestCase):
    def test_method_can_disable_els_reuse_and_warmups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            index_dir = root / "results" / "Tiny" / "index" / "Trie" / "index_files"
            block_index_dir = index_dir.parent / "block_index_files"
            query_dir = root / "data" / "Tiny" / "query_task"
            index_dir.mkdir(parents=True)
            block_index_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            (index_dir / "meta").write_text("index_name=Trie\n")
            (block_index_dir / "meta").write_text("index_name=TrieBlocks\n")
            query_bin = query_dir / "Tiny_query.bin"
            gt_file = query_dir / "gt.bin"
            query_bin.write_text("")
            gt_file.write_text("")

            captured = {}

            def fake_run_command(cmd, output_file, env=None):
                captured["env"] = env
                raw_dir = root / "results" / "Tiny" / "results" / "Trie" / "query_task_10_10_10" / "results"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "search_time_summary.csv").write_text(
                    "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n10,0,1,1\n"
                )

            cfg = {
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(build_dir),
                "search": {
                    "lsearch_start": 10,
                    "lsearch_step": 10,
                    "lsearch_end": 10,
                    "num_threads": 1,
                    "K": 1,
                    "num_repeats": 1,
                    "num_entry_points": 1,
                    "efs_start": 1,
                    "efs_step_slow": 1,
                    "efs_step_fast": 1,
                    "lsearch_threshold": 10,
                },
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {
                "name": "Trie",
                "index_name": "Trie",
                "entry_group_provider": "special_block_trie",
                "reuse_els": False,
                "warmup_special_block_trie": True,
            }
            with mock.patch.object(run_search_comparison, "run_command", fake_run_command):
                run_search_comparison.run_search(cfg, dataset_cfg, method, query_bin, gt_file, 1)

            self.assertEqual(captured["env"]["UNG_DISABLE_ELS_REUSE"], "1")
            self.assertEqual(captured["env"]["UNG_DISABLE_CPU_ELS_WARMUP"], "1")
            self.assertEqual(captured["env"]["UNG_DISABLE_SPECIAL_BLOCK_TRIE_WARMUP"], "1")

    def test_special_block_trie_can_defer_warmup_to_first_lsearch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            index_dir = root / "results" / "Tiny" / "index" / "Trie" / "index_files"
            block_index_dir = index_dir.parent / "block_index_files"
            query_dir = root / "data" / "Tiny" / "query_task"
            index_dir.mkdir(parents=True)
            block_index_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            (index_dir / "meta").write_text("index_name=Trie\n")
            (block_index_dir / "meta").write_text("index_name=TrieBlocks\n")
            query_bin = query_dir / "Tiny_query.bin"
            gt_file = query_dir / "gt.bin"
            query_bin.write_text("")
            gt_file.write_text("")

            captured = {}

            def fake_run_command(cmd, output_file, env=None):
                captured["env"] = env
                raw_dir = root / "results" / "Tiny" / "results" / "Trie" / "query_task_10_10_10" / "results"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "search_time_summary.csv").write_text(
                    "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n10,0,1,1\n"
                )

            cfg = {
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(build_dir),
                "search": {
                    "lsearch_start": 10,
                    "lsearch_step": 10,
                    "lsearch_end": 10,
                    "num_threads": 1,
                    "K": 1,
                    "num_repeats": 1,
                    "num_entry_points": 1,
                    "efs_start": 1,
                    "efs_step_slow": 1,
                    "efs_step_fast": 1,
                    "lsearch_threshold": 10,
                },
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {
                "name": "Trie",
                "index_name": "Trie",
                "entry_group_provider": "special_block_trie",
                "warmup_special_block_trie": False,
            }
            with mock.patch.object(run_search_comparison, "run_command", fake_run_command):
                run_search_comparison.run_search(cfg, dataset_cfg, method, query_bin, gt_file, 1)

            self.assertEqual(captured["env"]["UNG_DISABLE_SPECIAL_BLOCK_TRIE_WARMUP"], "1")

    def test_run_search_prefers_explicit_lsearch_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            index_dir = root / "results" / "Tiny" / "index" / "NaviX" / "index_files"
            query_dir = root / "data" / "Tiny" / "query_task"
            index_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            (index_dir / "meta").write_text("index_name=NaviX\n")
            query_bin = query_dir / "Tiny_query.bin"
            gt_file = query_dir / "gt.bin"
            query_bin.write_text("")
            gt_file.write_text("")

            captured = {}

            def fake_run_command(cmd, output_file, env=None):
                captured["cmd"] = cmd
                raw_dir = root / "results" / "Tiny" / "results" / "NaviX" / "query_task_100_100_4000" / "results"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "search_time_summary.csv").write_text("Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n100,0,1,1\n")

            cfg = {
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(build_dir),
                "search": {
                    "lsearch_start": 100,
                    "lsearch_step": 100,
                    "lsearch_end": 4000,
                    "lsearch_values": [100, 200, 1000, 2000, 3000, 4000],
                    "num_threads": 1,
                    "K": 1,
                    "num_repeats": 1,
                    "num_entry_points": 1,
                    "efs_start": 1,
                    "efs_step_slow": 1,
                    "efs_step_fast": 1,
                    "lsearch_threshold": 10,
                },
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {"name": "NaviX", "index_name": "NaviX", "entry_group_provider": "cpu_min_super_sets"}
            with mock.patch.object(run_search_comparison, "run_command", fake_run_command):
                run_search_comparison.run_search(cfg, dataset_cfg, method, query_bin, gt_file, 1)

            lsearch_idx = captured["cmd"].index("--Lsearch")
            end_idx = captured["cmd"].index("--lsearch_start")
            self.assertEqual(
                captured["cmd"][lsearch_idx + 1:end_idx],
                ["100", "200", "1000", "2000", "3000", "4000"],
            )

    def test_run_search_allows_method_to_override_lsearch_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            index_dir = root / "results" / "Tiny" / "index" / "NaviX" / "index_files"
            query_dir = root / "data" / "Tiny" / "query_task"
            index_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            (index_dir / "meta").write_text("index_name=NaviX\n")
            query_bin = query_dir / "Tiny_query.bin"
            gt_file = query_dir / "gt.bin"
            query_bin.write_text("")
            gt_file.write_text("")

            captured = {}

            def fake_run_command(cmd, output_file, env=None):
                captured["cmd"] = cmd
                captured["output_file"] = output_file
                raw_dir = (
                    root
                    / "results"
                    / "Tiny"
                    / "results"
                    / "NaviX"
                    / "query_task_100_250_1000"
                    / "results"
                )
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "search_time_summary.csv").write_text(
                    "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n100,0,1,1\n"
                )

            cfg = {
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(build_dir),
                "search": {
                    "lsearch_start": 100,
                    "lsearch_step": 100,
                    "lsearch_end": 1000,
                    "num_threads": 1,
                    "K": 1,
                    "num_repeats": 1,
                    "num_entry_points": 1,
                    "efs_start": 1,
                    "efs_step_slow": 1,
                    "efs_step_fast": 1,
                    "lsearch_threshold": 10,
                },
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {
                "name": "NaviX",
                "index_name": "NaviX",
                "entry_group_provider": "cpu_min_super_sets",
                "search": {"lsearch_step": 250},
            }
            with mock.patch.object(run_search_comparison, "run_command", fake_run_command):
                run_search_comparison.run_search(cfg, dataset_cfg, method, query_bin, gt_file, 1)

            lsearch_idx = captured["cmd"].index("--Lsearch")
            end_idx = captured["cmd"].index("--lsearch_start")
            step_idx = captured["cmd"].index("--lsearch_step")
            self.assertEqual(captured["cmd"][lsearch_idx + 1:end_idx], ["100", "350", "600", "850"])
            self.assertEqual(captured["cmd"][step_idx + 1], "250")
            self.assertIn("query_task_100_250_1000", str(captured["output_file"]))

    def test_run_search_uses_method_force_use_alg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            index_dir = root / "results" / "Tiny" / "index" / "NaviX" / "index_files"
            query_dir = root / "data" / "Tiny" / "query_task"
            index_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            (index_dir / "meta").write_text("index_name=NaviX\n")
            query_bin = query_dir / "Tiny_query.bin"
            gt_file = query_dir / "gt.bin"
            query_bin.write_text("")
            gt_file.write_text("")

            captured = {}
            def fake_run_command(cmd, output_file, env=None):
                captured["cmd"] = cmd
                raw_dir = root / "results" / "Tiny" / "results" / "NaviX" / "query_task_10_10_10" / "results"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "search_time_summary.csv").write_text("Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n10,0,1,1\n")

            cfg = {
                "data_root": str(root / "data"),
                "result_root": str(root / "results"),
                "build_dir": str(build_dir),
                "search": {
                    "lsearch_start": 10,
                    "lsearch_step": 10,
                    "lsearch_end": 10,
                    "num_threads": 1,
                    "K": 1,
                    "num_repeats": 1,
                    "num_entry_points": 1,
                    "efs_start": 1,
                    "efs_step_slow": 1,
                    "efs_step_fast": 1,
                    "lsearch_threshold": 10,
                },
            }
            dataset_cfg = {"dataset": "Tiny", "query_task": "query_task"}
            method = {"name": "NaviX", "index_name": "NaviX", "entry_group_provider": "cpu_min_super_sets", "force_use_alg": 5}
            with mock.patch.object(run_search_comparison, "run_command", fake_run_command):
                run_search_comparison.run_search(cfg, dataset_cfg, method, query_bin, gt_file, 1)

            idx = captured["cmd"].index("--force_use_alg")
            self.assertEqual(captured["cmd"][idx + 1], "5")


class MainLoggingTest(unittest.TestCase):
    def test_main_does_not_create_top_level_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            (build_dir / "apps").mkdir(parents=True)
            (build_dir / "apps" / "search_UNG_index").write_text("")
            cfg_path = root / "config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "build_dir": str(build_dir),
                        "data_root": str(root / "data"),
                        "result_root": str(root / "results"),
                        "search": {},
                        "datasets": [],
                        "methods": [],
                    }
                )
            )

            with mock.patch.object(sys, "argv", ["run_search_comparison.py", str(cfg_path)]):
                self.assertEqual(run_search_comparison.main(), 0)

            self.assertEqual(list(root.glob("search_comparison_*.log")), [])


class GroundTruthKIsolationTest(unittest.TestCase):
    def test_explicit_k20_file_is_used_instead_of_default_k10(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.bin"
            query.write_bytes(struct.pack("<II", 2, 1) + b"\0" * 8)
            default = root / "default"
            default.mkdir()
            (default / "Tiny_gt_labels_containment.bin").write_bytes(b"\0" * (2 * 10 * 8))
            k20 = root / "k20.bin"
            k20.write_bytes(b"\0" * (2 * 20 * 8))
            cfg = {"data_root": str(root), "build_dir": str(root), "search": {"K": 20}}
            row = {"dataset": "Tiny", "query_task": "q", "groundtruth_file": str(k20)}
            self.assertEqual(run_search_comparison.ensure_gt(cfg, row, query, default), k20)
            # The old K=10 payload must never be accepted for K=20.
            row.pop("groundtruth_file")
            with self.assertRaisesRegex(ValueError, "K=20"):
                run_search_comparison.ensure_gt(cfg, row, query, default)


if __name__ == "__main__":
    unittest.main()
