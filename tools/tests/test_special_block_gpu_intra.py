"""GPU integration check; set UNG_GPU_INTRA_TEST_BINARY and run via gpulock check."""

from array import array
from collections import Counter
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("UNG_GPU_INTRA_TEST_BINARY"),
                     "requires a CUDA build_special_block_index and gpulock check")
class SpecialBlockGpuIntraTest(unittest.TestCase):
    def test_serial_balanced_and_bounded_batches(self):
        binary = str(Path(os.environ["UNG_GPU_INTRA_TEST_BINARY"]).resolve())
        sizes = [33, 129, 257, 385, 513, 641, 769]
        total, dim = sum(sizes), 128
        with tempfile.TemporaryDirectory(prefix="ung_gpu_intra_") as tmp:
            root = Path(tmp)
            rng = random.Random(73)
            with (root / "vectors.bin").open("wb") as out:
                out.write(struct.pack("<II", total, dim))
                array("f", (rng.uniform(0.1, 0.9)
                            for _ in range(total * dim))).tofile(out)
            (root / "labels.txt").write_text("".join(
                f"{label}\n" * count for label, count in enumerate(sizes, 1)))

            for streams, point_limit, route, optimized in [
                    (1, 65536, 0, 1), (4, 65536, 0, 1), (4, 500, 0, 1),
                    (4, 65536, 1, 1), (1, 65536, 0, 0), (4, 65536, 0, 0)]:
                with self.subTest(streams=streams, point_limit=point_limit, route=route,
                                  optimized=optimized):
                    case = root / f"s{streams}_p{point_limit}_r{route}_t{optimized}"
                    case.mkdir()
                    env = {key: value for key, value in os.environ.items()
                           if not key.startswith("UNG_")}
                    env.update({
                        "UNG_BASE_GROUP_TOPOLOGY": "trie",
                        "UNG_SPECIAL_BLOCK_SKIP_TRIVIAL": "0",
                        "UNG_SPECIAL_BLOCK_GPU_INTRA": "1",
                        "UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT": "1",
                        "UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED": "1",
                        "UNG_SPECIAL_BLOCK_GPU_INTER": "0",
                        "UNG_SPECIAL_INTRA_COMPLETE_NX": "128",
                        "UNG_SPECIAL_INTRA_ROUTE": str(route),
                        "UNG_SPECIAL_INTRA_LARGE_NX": "256",
                        "UNG_TAGORE_BATCH_STREAMS": str(streams),
                        "UNG_SPECIAL_BLOCK_GPU_INTRA_BATCH_POINTS": str(point_limit),
                        "UNG_TAGORE_K": "128",
                        "UNG_TAGORE_ITER": "4",
                        "UNG_TAGORE_OPTIMIZED_TOPK": str(optimized),
                        "UNG_SPECIAL_EDGE_BINARY_ONLY": "1",
                    })
                    proc = subprocess.run([
                        binary, "--ung_index_path_prefix", str(root / "missing"),
                        "--base_bin_file", str(root / "vectors.bin"),
                        "--base_label_file", str(root / "labels.txt"),
                        "--block_index_path_prefix", str(case / "index"),
                        "--result_path_prefix", str(case / "results"),
                        "--data_type", "float", "--dist_fn", "L2",
                        "--num_threads", "2", "--min_points", "32",
                        "--max_degree", "32", "--num_cross_edges", "4",
                    ], env=env, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=120)
                    self.assertEqual(proc.returncode, 0, proc.stdout)
                    meta = dict(line.split("=", 1) for line in
                                (case / "index/meta").read_text().splitlines())
                    gpu_sizes = [n for n in sizes if n > (255 if route else 128)]
                    self.assertEqual(int(meta["gpu_intra_blocks"]), len(gpu_sizes))
                    self.assertEqual(int(meta["gpu_intra_points"]), sum(gpu_sizes))
                    self.assertEqual(meta["gpu_intra_fallback_blocks"], "0")
                    self.assertEqual(meta["cpu_intra_vamana_blocks"], "0")
                    self.assertEqual(meta["gpu_intra_optimized_topk"], str(optimized))
                    balanced = "[TagoreCuda][batch_balance]" in proc.stdout
                    self.assertEqual(balanced, streams > 1)
                    if streams > 1 and point_limit == 500:
                        self.assertGreater(proc.stdout.count("[GPU_INTRA_BATCH]"), 1)
                    for line in proc.stdout.splitlines():
                        if "[batch_balance]" in line or "[GPU_INTRA_BATCH]" in line:
                            print(f"s{streams}/p{point_limit}/r{route}: {line}")

                    # Check output ownership after worker results are reordered:
                    # all points have local edges, and targets stay in the same block.
                    data = (case / "index/special_edges.bin").read_bytes()
                    magic, version, kind, sources, edges = struct.unpack_from("<8sIIQQ", data)
                    self.assertEqual((magic, version, kind, sources),
                                     (b"SBCSR02\0", 2, 2, total))
                    offsets = struct.unpack_from(f"<{sources + 1}Q", data, 32)
                    self.assertEqual(offsets[-1], edges)
                    records = list(struct.iter_unpack("<II", data[32 + 8 * (sources + 1):]))
                    owners = []
                    for source in range(sources):
                        row = records[offsets[source]:offsets[source + 1]]
                        self.assertTrue(row)
                        self.assertLessEqual(len(row), 32)
                        self.assertEqual(len({target for target, _ in row}), len(row))
                        owner = row[0][1] >> 1
                        owners.append(owner)
                        for target, block_kind in row:
                            self.assertLess(target, sources)
                            self.assertNotEqual(target, source)
                            self.assertEqual(block_kind, owner << 1)
                    self.assertEqual(sorted(Counter(owners).values()), sorted(sizes))
                    for source in range(sources):
                        for target, _ in records[offsets[source]:offsets[source + 1]]:
                            self.assertEqual(owners[source], owners[target])


if __name__ == "__main__":
    unittest.main()
