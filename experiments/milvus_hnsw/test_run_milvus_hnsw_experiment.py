import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_milvus_hnsw_experiment.py")
SPEC = importlib.util.spec_from_file_location("run_milvus_hnsw_experiment", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class MilvusHnswRunnerTest(unittest.TestCase):
    def test_current_knowhere_defaults(self):
        build = runner.build_config({}, {}, {})
        self.assertEqual(build["M"], 30)
        self.assertEqual(build["ef_construction"], 360)
        self.assertEqual(build["add_batch_size"], 0)
        search = runner.search_config({}, {}, {})
        self.assertFalse(search["disable_fallback_brute_force"])

    def test_groundtruth_pair_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pairs.bin"
            target = root / "ids.bin"
            with source.open("wb") as stream:
                stream.write(struct.pack("<IfIf", 7, 0.25, 11, 0.5))
            runner.prepare_groundtruth(source, target, topk=2, query_count=1)
            self.assertEqual(struct.unpack("<ii", target.read_bytes()), (7, 11))

    def test_manifest_controls_index_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.fvecs"
            labels = root / "labels.txt"
            base.write_bytes(struct.pack("<iff", 2, 1.0, 2.0))
            labels.write_text("3\n", encoding="utf-8")
            cfg = {"data_root": str(root), "result_root": str(root / "results")}
            dataset = {
                "dataset": "Tiny", "query_task": "q",
                "base_fvecs": str(base), "base_labels": str(labels),
            }
            method = {"name": "milvus_hnsw", "index_name": "Milvus_HNSW_default"}
            paths = runner.resolve_paths(cfg, dataset, method)
            self.assertEqual(
                paths["adapter_groundtruth"],
                root / "results" / "Tiny" / "results" / "milvus_hnsw" /
                "q_100_100_2000" / "results" / "Tiny_milvus_hnsw_gt_ids.bin",
            )
            expected = runner.expected_manifest(cfg, dataset, method, paths)
            for key in ["index_file", "filter_file"]:
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_bytes(b"index")
            paths["manifest"].write_text(json.dumps(expected), encoding="utf-8")
            self.assertTrue(runner.index_is_current(paths, expected))
            changed = json.loads(json.dumps(expected))
            changed["build"]["M"] = 64
            self.assertFalse(runner.index_is_current(paths, changed))


if __name__ == "__main__":
    unittest.main()
