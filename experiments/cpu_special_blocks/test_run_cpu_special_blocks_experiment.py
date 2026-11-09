#!/usr/bin/env python3

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "run_cpu_special_blocks_experiment.sh"


class CpuSpecialBlocksRunnerTest(unittest.TestCase):
    def test_complete_cross_block_index_skips_compile_and_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_root = root / "results"
            index_root = (
                result_root / "Tiny" / "index" /
                "Trie_block_100_M_64_cross_block_4"
            )
            index_dir = index_root / "index_files"
            block_dir = index_root / "block_index_files"
            index_dir.mkdir(parents=True)
            block_dir.mkdir(parents=True)

            (index_dir / "meta").write_text(
                "base_group_topology=trie\n", encoding="utf-8")
            for name in ("vecs.bin", "labels.txt", "group_id_to_label_set",
                         "group_id_to_range"):
                (index_dir / name).write_bytes(b"complete")
            (block_dir / "meta").write_text(
                "special_block_min_points=100\n"
                "special_block_max_degree=64\n"
                "special_block_num_cross_edges=4\n",
                encoding="utf-8",
            )
            for name in ("special_edges.bin", "special_blocks.bin",
                         "special_block_trie.bin"):
                (block_dir / name).write_bytes(b"complete")

            fake_bin = root / "bin"
            fake_bin.mkdir()
            cmake_log = root / "cmake.log"
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${FAKE_CMAKE_LOG}\"\n"
                "exit 99\n",
                encoding="utf-8",
            )
            fake_cmake.chmod(0o755)

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "reuse_complete_index_test",
                        "output_layout": "index_by_dataset",
                        "index_name": "Trie_block",
                        "data_root": str(root / "missing-data"),
                        "result_root": str(result_root),
                        "build_dir": str(root / "build"),
                        "build_ung": True,
                        "require_ung_index": True,
                        "required_base_topology": "trie",
                        "special_block_num_cross_edges_values": [4],
                        "datasets": ["Tiny"],
                        "build": {},
                        "ung_env": {
                            "UNG_SPECIAL_BLOCKS": "1",
                            "UNG_SPECIAL_BLOCK_MIN_POINTS": "100",
                            "UNG_SPECIAL_BLOCK_MAX_DEGREE": "64",
                            "UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES": "4",
                        },
                        "index_name_params": [],
                    }
                ),
                encoding="utf-8",
            )
            env = {
                **__import__("os").environ,
                "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
                "FAKE_CMAKE_LOG": str(cmake_log),
            }
            completed = subprocess.run(
                [str(RUNNER), str(config_path)],
                check=True,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertIn("complete index exists; skipping build",
                          completed.stdout)
            self.assertFalse(cmake_log.exists())

    def test_keeps_dataset_others_but_does_not_write_run_logs_or_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            build_dir = root / "build"
            config_dir = root / "config"
            fake_bin = root / "bin"
            dataset_dir = data_root / "Tiny"
            dataset_dir.mkdir(parents=True)
            build_app = build_dir / "apps" / "build_UNG_index"
            build_app.parent.mkdir(parents=True)

            (dataset_dir / "Tiny_base.bin").write_bytes(b"tiny")
            (dataset_dir / "Tiny_base_labels.txt").write_text("0\n", encoding="utf-8")
            build_app.write_text(
                "#!/usr/bin/env bash\n"
                "set -e\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --index_path_prefix) index_dir=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$index_dir\"\n"
                "printf 'num_points=1\\n' > \"${index_dir}/meta\"\n",
                encoding="utf-8",
            )
            build_app.chmod(0o755)

            fake_bin.mkdir()
            cmake_log = root / "cmake.log"
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\n' \"$*\" >> \"${FAKE_CMAKE_LOG}\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_cmake.chmod(0o755)

            config_dir.mkdir()
            config_path = config_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "cpu_special_blocks_test",
                        "output_layout": "index_by_dataset",
                        "index_name": "UNG_test",
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "build_dir": str(build_dir),
                        "datasets": ["Tiny"],
                        "build": {},
                        "ung_env": {},
                    }
                ),
                encoding="utf-8",
            )

            env = {
                **__import__("os").environ,
                "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
                "FAKE_CMAKE_LOG": str(cmake_log),
            }
            subprocess.run([str(RUNNER), str(config_path)], check=True, cwd=REPO_ROOT, env=env)

            cmake_calls = cmake_log.read_text(encoding="utf-8")
            self.assertIn("-S", cmake_calls)
            self.assertIn("--build", cmake_calls)
            self.assertIn("--target build_UNG_index", cmake_calls)

            out_dir = result_root / "Tiny" / "index" / "UNG_test"
            self.assertTrue((out_dir / "others" / "build.log").exists())
            self.assertTrue((out_dir / "others" / "time.txt").exists())
            self.assertFalse(list(config_dir.glob("*.log")))
            self.assertFalse(list(result_root.rglob("summary.csv")))

    def test_reuses_required_external_trie_index_for_multilevel_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            build_dir = root / "build"
            source_index = root / "base_trie" / "index_files"
            fake_bin = root / "bin"
            dataset_dir = data_root / "Tiny"
            dataset_dir.mkdir(parents=True)
            source_index.mkdir(parents=True)

            (dataset_dir / "Tiny_base.bin").write_bytes(b"tiny")
            (dataset_dir / "Tiny_base_labels.txt").write_text("0\n", encoding="utf-8")
            (source_index / "meta").write_text(
                "base_group_topology=trie\n", encoding="utf-8")
            for name in ("vecs.bin", "labels.txt", "group_id_to_label_set",
                         "group_id_to_range"):
                (source_index / name).write_bytes(b"source")

            block_app = build_dir / "apps" / "build_special_block_index"
            block_app.parent.mkdir(parents=True)
            block_app.write_text(
                "#!/usr/bin/env bash\n"
                "set -e\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --ung_index_path_prefix) source_dir=\"$2\"; shift 2 ;;\n"
                "    --block_index_path_prefix) block_dir=\"$2\"; shift 2 ;;\n"
                "    --result_path_prefix) result_dir=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$block_dir\" \"$result_dir\"\n"
                "printf '%s\\n' \"$source_dir\" > \"${block_dir}/source_path.txt\"\n"
                "printf '%s\\n' \"${UNG_HIERARCHY_LAYERS:-}\" > \"${block_dir}/layers.txt\"\n",
                encoding="utf-8",
            )
            block_app.chmod(0o755)

            fake_bin.mkdir()
            fake_nvcc = fake_bin / "nvcc"
            fake_nvcc.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_nvcc.chmod(0o755)
            cmake_log = root / "cmake.log"
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${FAKE_CMAKE_LOG}\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_cmake.chmod(0o755)

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "multilevel_trie_test",
                        "output_layout": "index_by_dataset",
                        "index_name": "Trie_multilevel",
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "build_dir": str(build_dir),
                        "build_ung": False,
                        "ung_index_path_prefix": str(source_index),
                        "require_ung_index": True,
                        "required_base_topology": "trie",
                        "cuda": {
                            "enabled": True,
                            "required": False,
                            "compiler": str(fake_nvcc),
                            "architectures": "89",
                        },
                        "datasets": ["Tiny"],
                        "build": {},
                        "ung_env": {
                            "UNG_SPECIAL_BLOCKS": "1",
                            "UNG_BASE_GROUP_TOPOLOGY": "trie",
                            "UNG_HIERARCHY_LAYERS": "1000:trie,16000:trie",
                            "UNG_SPECIAL_BLOCK_MIN_POINTS": "1000",
                        },
                    }
                ),
                encoding="utf-8",
            )

            env = {
                **__import__("os").environ,
                "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
                "FAKE_CMAKE_LOG": str(cmake_log),
            }
            subprocess.run([str(RUNNER), str(config_path)], check=True,
                           cwd=REPO_ROOT, env=env)

            cmake_calls = cmake_log.read_text(encoding="utf-8")
            self.assertIn("--target build_special_block_index", cmake_calls)
            self.assertNotIn("--target build_UNG_index", cmake_calls)
            self.assertIn("-DUNG_ENABLE_CUDA=ON", cmake_calls)
            self.assertIn(f"-DCMAKE_CUDA_COMPILER={fake_nvcc}", cmake_calls)
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89", cmake_calls)
            block_dir = result_root / "Tiny" / "index" / "Trie_multilevel" / "block_index_files"
            self.assertEqual(
                (block_dir / "source_path.txt").read_text(encoding="utf-8").strip(),
                str(source_index) + "/",
            )
            self.assertEqual(
                (block_dir / "layers.txt").read_text(encoding="utf-8").strip(),
                "1000:trie,16000:trie",
            )

    def test_single_level_min_points_sweep_uses_separate_index_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            build_dir = root / "build"
            fake_bin = root / "bin"
            dataset_dir = data_root / "Tiny"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "Tiny_base.bin").write_bytes(b"tiny")
            (dataset_dir / "Tiny_base_labels.txt").write_text(
                "0\n", encoding="utf-8")

            block_app = build_dir / "apps" / "build_special_block_index"
            block_app.parent.mkdir(parents=True)
            block_app.write_text(
                "#!/usr/bin/env bash\n"
                "set -e\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --block_index_path_prefix) block_dir=\"$2\"; shift 2 ;;\n"
                "    --min_points) min_points=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$block_dir\"\n"
                "printf '%s:%s\\n' \"$min_points\" "
                "\"${UNG_SPECIAL_BLOCK_MIN_POINTS:-}\" > "
                "\"${block_dir}/min_points.txt\"\n",
                encoding="utf-8",
            )
            block_app.chmod(0o755)

            fake_bin.mkdir()
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_cmake.chmod(0o755)

            values = [100, 1000, 5000, 10000]
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "single_level_sweep_test",
                        "output_layout": "index_by_dataset",
                        "index_name": "Trie_block",
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "build_dir": str(build_dir),
                        "build_ung": False,
                        "special_block_min_points_values": values,
                        "datasets": ["Tiny"],
                        "build": {},
                        "ung_env": {
                            "UNG_SPECIAL_BLOCKS": "1",
                            "UNG_SPECIAL_BLOCK_MIN_POINTS": "999",
                        },
                        "index_name_params": [],
                    }
                ),
                encoding="utf-8",
            )

            env = {
                **__import__("os").environ,
                "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
            }
            subprocess.run([str(RUNNER), str(config_path)], check=True,
                           cwd=REPO_ROOT, env=env)

            index_root = result_root / "Tiny" / "index"
            self.assertEqual(
                sorted(path.name for path in index_root.iterdir()),
                ["Trie_block_100", "Trie_block_1000", "Trie_block_10000",
                 "Trie_block_5000"],
            )
            for value in values:
                value_file = (
                    index_root / f"Trie_block_{value}" /
                    "block_index_files" / "min_points.txt"
                )
                self.assertEqual(value_file.read_text(encoding="utf-8").strip(),
                                 f"{value}:{value}")

    def test_block_degree_sweep_uses_m_degree_index_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            build_dir = root / "build"
            fake_bin = root / "bin"
            dataset_dir = data_root / "Tiny"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "Tiny_base.bin").write_bytes(b"tiny")
            (dataset_dir / "Tiny_base_labels.txt").write_text(
                "0\n", encoding="utf-8")

            block_app = build_dir / "apps" / "build_special_block_index"
            block_app.parent.mkdir(parents=True)
            block_app.write_text(
                "#!/usr/bin/env bash\n"
                "set -e\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --block_index_path_prefix) block_dir=\"$2\"; shift 2 ;;\n"
                "    --max_degree) max_degree=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$block_dir\"\n"
                "printf '%s\\n' \"$max_degree\" > "
                "\"$block_dir/max_degree.txt\"\n",
                encoding="utf-8",
            )
            block_app.chmod(0o755)

            fake_bin.mkdir()
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_cmake.chmod(0o755)

            values = [16, 32, 64]
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "degree_sweep_test",
                        "output_layout": "index_by_dataset",
                        "index_name": "Trie_block",
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "build_dir": str(build_dir),
                        "build_ung": False,
                        "special_block_max_degree_values": values,
                        "datasets": ["Tiny"],
                        "build": {},
                        "ung_env": {
                            "UNG_SPECIAL_BLOCKS": "1",
                            "UNG_SPECIAL_BLOCK_MIN_POINTS": "100",
                            "UNG_SPECIAL_BLOCK_MAX_DEGREE": "999",
                        },
                        "index_name_params": [],
                    }
                ),
                encoding="utf-8",
            )

            env = {
                **__import__("os").environ,
                "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
            }
            subprocess.run([str(RUNNER), str(config_path)], check=True,
                           cwd=REPO_ROOT, env=env)

            index_root = result_root / "Tiny" / "index"
            self.assertEqual(
                sorted(path.name for path in index_root.iterdir()),
                ["Trie_block_100_M_16", "Trie_block_100_M_32",
                 "Trie_block_100_M_64"],
            )
            for value in values:
                value_file = (
                    index_root / f"Trie_block_100_M_{value}" /
                    "block_index_files" / "max_degree.txt"
                )
                self.assertEqual(value_file.read_text(encoding="utf-8").strip(),
                                 str(value))

    def test_block_cross_edge_sweep_uses_degree_and_cross_block_index_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            build_dir = root / "build"
            fake_bin = root / "bin"
            dataset_dir = data_root / "Tiny"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "Tiny_base.bin").write_bytes(b"tiny")
            (dataset_dir / "Tiny_base_labels.txt").write_text(
                "0\n", encoding="utf-8")

            block_app = build_dir / "apps" / "build_special_block_index"
            block_app.parent.mkdir(parents=True)
            block_app.write_text(
                "#!/usr/bin/env bash\n"
                "set -e\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --block_index_path_prefix) block_dir=\"$2\"; shift 2 ;;\n"
                "    --max_degree) max_degree=\"$2\"; shift 2 ;;\n"
                "    --num_cross_edges) cross_edges=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$block_dir\"\n"
                "printf '%s:%s\\n' \"$max_degree\" \"$cross_edges\" > "
                "\"$block_dir/block_params.txt\"\n",
                encoding="utf-8",
            )
            block_app.chmod(0o755)

            fake_bin.mkdir()
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_cmake.chmod(0o755)

            values = [2, 4, 6]
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "cross_edge_sweep_test",
                        "output_layout": "index_by_dataset",
                        "index_name": "Trie_block",
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "build_dir": str(build_dir),
                        "build_ung": False,
                        "special_block_num_cross_edges_values": values,
                        "datasets": ["Tiny"],
                        "build": {},
                        "ung_env": {
                            "UNG_SPECIAL_BLOCKS": "1",
                            "UNG_SPECIAL_BLOCK_MIN_POINTS": "100",
                            "UNG_SPECIAL_BLOCK_MAX_DEGREE": "64",
                            "UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES": "999",
                        },
                        "index_name_params": [],
                    }
                ),
                encoding="utf-8",
            )

            env = {
                **__import__("os").environ,
                "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
            }
            subprocess.run([str(RUNNER), str(config_path)], check=True,
                           cwd=REPO_ROOT, env=env)

            index_root = result_root / "Tiny" / "index"
            self.assertEqual(
                sorted(path.name for path in index_root.iterdir()),
                ["Trie_block_100_M_64_cross_block_2",
                 "Trie_block_100_M_64_cross_block_4",
                 "Trie_block_100_M_64_cross_block_6"],
            )
            for value in values:
                value_file = (
                    index_root /
                    f"Trie_block_100_M_64_cross_block_{value}" /
                    "block_index_files" / "block_params.txt"
                )
                self.assertEqual(value_file.read_text(encoding="utf-8").strip(),
                                 f"64:{value}")

    def test_cross_edge_sweep_configs_fix_threshold_and_degree(self):
        for dataset in ("Amazon", "VImg", "Music"):
            with self.subTest(dataset=dataset):
                config_path = (
                    REPO_ROOT / "experiments" / "cpu_special_blocks" /
                    f"{dataset}_config_Trie_block_cross_block_sweep.json"
                )
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["datasets"], [dataset])
                self.assertEqual(config["special_block_num_cross_edges_values"],
                                 [2, 4, 6])
                self.assertNotIn("special_block_min_points_values", config)
                self.assertNotIn("special_block_max_degree_values", config)
                env = config["ung_env"]
                self.assertEqual(env["UNG_SPECIAL_BLOCK_MIN_POINTS"], "500")
                self.assertEqual(env["UNG_SPECIAL_BLOCK_MAX_DEGREE"], "32")
                self.assertEqual(env["UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES"], "4")

    def test_campaign_configs_fix_threshold_degree_and_cross_edges(self):
        for dataset in (
            "Amazon", "BReviews", "Laion", "Music",
            "Tiktok", "VImg", "Genome", "Reviews",
        ):
            with self.subTest(dataset=dataset):
                config_path = (
                    REPO_ROOT / "experiments" / "cpu_special_blocks" /
                    f"{dataset}_config_Trie_block.json"
                )
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["datasets"], [dataset])
                self.assertEqual(config["index_name"], "Trie_block")
                self.assertEqual(
                    config["special_block_min_points_values"],
                    [500],
                )
                self.assertEqual(
                    config["special_block_max_degree_values"],
                    [32],
                )
                self.assertEqual(
                    config["special_block_num_cross_edges_values"],
                    [4],
                )
                self.assertEqual(config["index_name_params"], [])
                env = config["ung_env"]
                self.assertEqual(env["UNG_SPECIAL_BLOCK_MIN_POINTS"], "500")
                self.assertEqual(env["UNG_SPECIAL_BLOCK_MAX_DEGREE"], "32")
                self.assertEqual(env["UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES"], "4")
                self.assertNotIn("UNG_HIERARCHY_LAYERS", env)
                self.assertNotIn("UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS", env)
                self.assertEqual(env["UNG_BASE_GROUP_TOPOLOGY"], "trie")
                self.assertEqual(env["UNG_SPECIAL_BLOCK_PARTITION"], "trie")

    def test_degree_sweep_configs_fix_threshold_and_cross_edges(self):
        for dataset in ("Amazon", "Music", "VImg"):
            with self.subTest(dataset=dataset):
                config_path = (
                    REPO_ROOT / "experiments" / "cpu_special_blocks" /
                    f"{dataset}_config_Trie_block_M_sweep.json"
                )
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["datasets"], [dataset])
                self.assertEqual(config["index_name"], "Trie_block")
                self.assertEqual(config["special_block_max_degree_values"],
                                 [16, 32, 64])
                self.assertNotIn("special_block_min_points_values", config)
                self.assertNotIn("special_block_num_cross_edges_values", config)
                env = config["ung_env"]
                self.assertEqual(env["UNG_SPECIAL_BLOCK_MIN_POINTS"], "500")
                self.assertEqual(env["UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES"], "4")
                self.assertEqual(env["UNG_TAGORE_K"], "128")

    def test_amazon_multilevel_config_self_builds_trie_and_requires_gpu(self):
        config_path = (
            REPO_ROOT / "experiments" / "cpu_special_blocks" /
            "Amazon_config_Multilevel_Trie_block.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertTrue(config["build_ung"])
        self.assertEqual(
            config["index_name"],
            "Trie_multilevel_gpu_block_sourceexact_complete32_"
            "cross_edges-4-LAYERS-3-8192-13072",
        )
        self.assertEqual(config["index_name_params"], [])
        self.assertNotIn("ung_index_path_prefix", config)
        self.assertEqual(config["required_base_topology"], "trie")
        self.assertEqual(config["build"]["num_threads"], 60)
        self.assertTrue(config["cuda"]["enabled"])
        self.assertTrue(config["cuda"]["required"])
        self.assertEqual(config["cuda"]["architectures"], "89")
        env = config["ung_env"]
        self.assertEqual(env["UNG_BASE_GROUP_TOPOLOGY"], "trie")
        self.assertEqual(env["UNG_HIERARCHY_LAYERS"], "8192:trie,13072:trie")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_MIN_POINTS"], "8192")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS"], "13072")
        self.assertEqual(env["UNG_CROSS_EDGE_IMPL"], "4")
        self.assertEqual(env["UNG_CPU_TARGET_EXACT_MAX_NX"], "1000")
        self.assertEqual(env["UNG_CPU_HYBRID_SCAN_MAX_WORK"], "10000")
        self.assertEqual(env["UNG_TRIE_CROSS_SOURCE_EXACT"], "1")
        self.assertEqual(env["UNG_ADDITIONAL_EDGES_IMPL"], "2")
        self.assertEqual(env["UNG_GROUP_GRAPH_COMPLETE_NX"], "32")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_GPU_INTRA"], "1")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT"], "1")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED"], "1")


if __name__ == "__main__":
    unittest.main()
