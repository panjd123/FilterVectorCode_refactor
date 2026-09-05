import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


def write_fvecs(path: Path, vectors):
    with path.open("wb") as f:
        for vector in vectors:
            f.write(struct.pack("<i", len(vector)))
            f.write(struct.pack("<" + "f" * len(vector), *vector))


def read_query_labels(path: Path):
    labels = []
    for line in path.read_text().splitlines():
        if line.strip():
            labels.append(tuple(int(part) for part in line.split(",") if part))
    return labels


class VariableSubBaseParentRangeTest(unittest.TestCase):
    def test_samples_only_from_configured_parent_label_range(self):
        repo_root = Path(__file__).resolve().parents[2]
        tool = repo_root / "build_ung_rel" / "tools" / "generate_mixed_queries"
        self.assertTrue(tool.is_file(), f"generate_mixed_queries not found: {tool}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "Tiny_base_labels.txt"
            vectors_path = tmp_path / "Tiny_base.fvecs"

            base_labels = list(range(100, 120))
            labels_path.write_text("".join(f"{label}\n" for label in base_labels))
            write_fvecs(vectors_path, [[float(i), float(i + 1)] for i in range(20)])

            first_70_labels = self.run_generator(
                tool, labels_path, vectors_path, tmp_path / "front", 0.0, 0.7
            )
            last_30_labels = self.run_generator(
                tool, labels_path, vectors_path, tmp_path / "back", 0.7, 1.0
            )

            self.assertEqual(60, len(first_70_labels))
            self.assertEqual(60, len(last_30_labels))
            self.assertTrue(
                all(len(labels) == 1 and labels[0] in base_labels[:14] for labels in first_70_labels)
            )
            self.assertTrue(
                all(len(labels) == 1 and labels[0] in base_labels[14:] for labels in last_30_labels)
            )

    def run_generator(self, tool, labels_path, vectors_path, output_prefix, start, end):
        output_labels = output_prefix.with_suffix(".labels.txt")
        output_vectors = output_prefix.with_suffix(".fvecs")
        subprocess.run(
            [
                str(tool),
                "--mode",
                "variable_sub_base",
                "--input_file",
                str(labels_path),
                "--output_file",
                str(output_labels),
                "--base_vectors_file",
                str(vectors_path),
                "--output_vectors_file",
                str(output_vectors),
                "--num_points",
                "60",
                "--min-query-length",
                "1",
                "--max-query-length",
                "1",
                "--K",
                "1",
                "--max-coverage",
                "20",
                "--min-children",
                "0",
                "--parent-range-start",
                str(start),
                "--parent-range-end",
                str(end),
            ],
            check=True,
        )
        return read_query_labels(output_labels)

    def test_batch_average_selectivity_can_mix_discrete_coverage_levels(self):
        repo_root = Path(__file__).resolve().parents[2]
        tool = repo_root / "build_ung_rel" / "tools" / "generate_mixed_queries"
        self.assertTrue(tool.is_file(), f"generate_mixed_queries not found: {tool}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "Tiny_base_labels.txt"
            vectors_path = tmp_path / "Tiny_base.fvecs"
            output_labels = tmp_path / "average.labels.txt"
            output_vectors = tmp_path / "average.fvecs"

            labels_path.write_text("".join(["1\n"] * 8 + ["2\n"] * 2))
            write_fvecs(vectors_path, [[float(i), float(i + 1)] for i in range(10)])

            subprocess.run(
                [
                    str(tool),
                    "--mode",
                    "variable_sub_base",
                    "--input_file",
                    str(labels_path),
                    "--output_file",
                    str(output_labels),
                    "--base_vectors_file",
                    str(vectors_path),
                    "--output_vectors_file",
                    str(output_vectors),
                    "--num_points",
                    "10",
                    "--min-query-length",
                    "1",
                    "--max-query-length",
                    "1",
                    # No individual query can satisfy this range. Batch-average
                    # mode must intentionally ignore the per-query bounds.
                    "--K",
                    "4",
                    "--max-coverage",
                    "6",
                    "--min-children",
                    "0",
                    "--target-average-selectivity",
                    "0.5",
                    "--average-selectivity-tolerance",
                    "0",
                    "--average-candidate-pool-size",
                    "0",
                ],
                check=True,
            )

            query_labels = read_query_labels(output_labels)
            self.assertEqual(10, len(query_labels))
            coverages = {1: 8, 2: 2}
            actual_selectivity = sum(coverages[labels[0]] for labels in query_labels) / 100
            self.assertEqual(0.5, actual_selectivity)

    def test_batch_average_selectivity_prefers_distinct_label_sets(self):
        repo_root = Path(__file__).resolve().parents[2]
        tool = repo_root / "build_ung_rel" / "tools" / "generate_mixed_queries"
        self.assertTrue(tool.is_file(), f"generate_mixed_queries not found: {tool}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "Tiny_base_labels.txt"
            vectors_path = tmp_path / "Tiny_base.fvecs"
            output_labels = tmp_path / "diverse.labels.txt"
            output_vectors = tmp_path / "diverse.fvecs"

            # Label coverages are 10, 9, ..., 1. Six distinct labels can
            # therefore have an exact mean coverage of 5.5.
            labels_path.write_text(
                "".join(",".join(str(label) for label in range(1, 11 - row)) + "\n" for row in range(10))
            )
            write_fvecs(vectors_path, [[float(i), float(i + 1)] for i in range(10)])

            subprocess.run(
                [
                    str(tool),
                    "--mode",
                    "variable_sub_base",
                    "--input_file",
                    str(labels_path),
                    "--output_file",
                    str(output_labels),
                    "--base_vectors_file",
                    str(vectors_path),
                    "--output_vectors_file",
                    str(output_vectors),
                    "--num_points",
                    "6",
                    "--min-query-length",
                    "1",
                    "--max-query-length",
                    "1",
                    "--K",
                    "1",
                    "--max-coverage",
                    "10",
                    "--min-children",
                    "0",
                    "--target-average-selectivity",
                    "0.55",
                    "--average-selectivity-tolerance",
                    "0",
                    "--average-candidate-pool-size",
                    "0",
                ],
                check=True,
            )

            query_labels = read_query_labels(output_labels)
            self.assertEqual(6, len(query_labels))
            self.assertEqual(6, len(set(query_labels)))
            self.assertEqual(33, sum(11 - labels[0] for labels in query_labels))


if __name__ == "__main__":
    unittest.main()
