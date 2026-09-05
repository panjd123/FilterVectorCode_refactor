#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_search_comparison


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


class RunCommandLoggingTest(unittest.TestCase):
    def test_run_command_can_stream_without_writing_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "command.log"

            run_search_comparison.run_command(
                [sys.executable, "-c", "print('child output')"],
                None,
            )

            self.assertFalse(log_path.exists())


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


if __name__ == "__main__":
    unittest.main()
