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


def find_generator_tool(repo_root: Path) -> Path:
    for build_dir in ("build_ung_multilevel_trie_gpu", "build_ung_rel"):
        tool = repo_root / build_dir / "tools" / "generate_mixed_queries"
        if tool.is_file():
            return tool
    return repo_root / "build_ung_multilevel_trie_gpu" / "tools" / "generate_mixed_queries"


class VariableSubBaseParentRangeTest(unittest.TestCase):
    def test_samples_only_from_configured_parent_label_range(self):
        repo_root = Path(__file__).resolve().parents[2]
        tool = find_generator_tool(repo_root)
        self.assertTrue(tool.is_file(), f"generate_mixed_queries not found: {tool}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "Tiny_base_labels.txt"
            vectors_path = tmp_path / "Tiny_base.fvecs"

            label_universe = list(range(100, 120))
            base_labels = [label for label in label_universe for _ in range(20)]
            labels_path.write_text("".join(f"{label}\n" for label in base_labels))
            write_fvecs(
                vectors_path,
                [[float(i), float(i + 1)] for i in range(len(base_labels))],
            )

            first_70_labels = self.run_generator(
                tool, labels_path, vectors_path, tmp_path / "front", 0.0, 0.7
            )
            last_30_labels = self.run_generator(
                tool, labels_path, vectors_path, tmp_path / "back", 0.7, 1.0
            )

            self.assertEqual(60, len(first_70_labels))
            self.assertEqual(60, len(last_30_labels))
            self.assertTrue(
                all(len(labels) == 1 and labels[0] in label_universe[:14] for labels in first_70_labels)
            )
            self.assertTrue(
                all(len(labels) == 1 and labels[0] in label_universe[14:] for labels in last_30_labels)
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

    def test_fixed_selectivity_range_uses_candidate_coverage_and_boundaries(self):
        repo_root = Path(__file__).resolve().parents[2]
        tool = find_generator_tool(repo_root)
        self.assertTrue(tool.is_file(), f"generate_mixed_queries not found: {tool}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "Tiny_base_labels.txt"
            vectors_path = tmp_path / "Tiny_base.fvecs"
            output_labels = tmp_path / "range.labels.txt"
            output_vectors = tmp_path / "range.fvecs"

            base_labels = []
            for row in range(100):
                labels = [1]
                if row < 60:
                    labels.append(2)
                if row < 40:
                    labels.append(3)
                if row < 20:
                    labels.append(4)
                base_labels.append(",".join(map(str, labels)))
            labels_path.write_text("\n".join(base_labels) + "\n")
            write_fvecs(vectors_path, [[float(i), float(i + 1)] for i in range(100)])

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
                    "5",
                    "--min-query-length",
                    "1",
                    "--max-query-length",
                    "1",
                    "--K",
                    "1",
                    "--min-groundtruth",
                    "20",
                    "--min-children",
                    "0",
                    "--min-selectivity",
                    "0.2",
                    "--max-selectivity",
                    "0.4",
                    "--selectivity-candidate-pool-size",
                    "0",
                    "--random-seed",
                    "7",
                ],
                check=True,
            )

            query_labels = read_query_labels(output_labels)
            self.assertEqual(5, len(query_labels))
            # The lower bound is inclusive (label 4 has 20% coverage), while
            # the upper bound is exclusive (label 3 has exactly 40%).
            self.assertEqual({(4,)}, set(query_labels))

    def test_fixed_high_range_exhaustively_finds_multilabel_candidates(self):
        repo_root = Path(__file__).resolve().parents[2]
        tool = find_generator_tool(repo_root)
        self.assertTrue(tool.is_file(), f"generate_mixed_queries not found: {tool}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "Tiny_base_labels.txt"
            vectors_path = tmp_path / "Tiny_base.fvecs"
            output_labels = tmp_path / "high.labels.txt"
            output_vectors = tmp_path / "high.fvecs"

            base_labels = []
            for row in range(100):
                labels = [1, 2]
                if row < 60:
                    labels.append(3)
                base_labels.append(",".join(map(str, labels)))
            labels_path.write_text("\n".join(base_labels) + "\n")
            write_fvecs(vectors_path, [[float(i), float(i + 1)] for i in range(100)])

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
                    "2",
                    "--max-query-length",
                    "2",
                    "--K",
                    "1",
                    "--min-groundtruth",
                    "20",
                    "--min-children",
                    "0",
                    "--min-selectivity",
                    "0.4",
                    "--max-selectivity",
                    "1.0",
                    "--selectivity-candidate-pool-size",
                    "0",
                    "--random-seed",
                    "9",
                ],
                check=True,
            )

            query_labels = read_query_labels(output_labels)
            self.assertEqual(6, len(query_labels))
            self.assertEqual({(1, 2), (1, 3), (2, 3)}, set(query_labels))


    def test_selectivity_extremes_are_reserved_outside_preferred_range(self):
        repo_root = Path(__file__).resolve().parents[2]
        tool = find_generator_tool(repo_root)
        self.assertTrue(tool.is_file(), f"generate_mixed_queries not found: {tool}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "Tiny_base_labels.txt"
            vectors_path = tmp_path / "Tiny_base.fvecs"
            output_labels = tmp_path / "extremes.labels.txt"
            output_vectors = tmp_path / "extremes.fvecs"

            base_labels = []
            for row in range(100):
                labels = [1]
                if row < 80:
                    labels.append(2)
                if row < 60:
                    labels.append(3)
                if row < 40:
                    labels.append(4)
                if row < 20:
                    labels.append(5)
                base_labels.append(",".join(map(str, labels)))
            labels_path.write_text("\n".join(base_labels) + "\n")
            write_fvecs(vectors_path, [[float(i), float(i + 1)] for i in range(100)])

            subprocess.run(
                [
                    str(tool),
                    "--mode", "variable_sub_base",
                    "--input_file", str(labels_path),
                    "--output_file", str(output_labels),
                    "--base_vectors_file", str(vectors_path),
                    "--output_vectors_file", str(output_vectors),
                    "--num_points", "6",
                    "--min-query-length", "1",
                    "--max-query-length", "1",
                    "--K", "1",
                    "--min-groundtruth", "20",
                    "--min-children", "0",
                    "--min-selectivity", "0.2",
                    "--max-selectivity", "0.9",
                    "--preferred-selectivity-min", "0.4",
                    "--preferred-selectivity-max", "0.7",
                    "--include-selectivity-extremes", "true",
                    "--selectivity-candidate-pool-size", "0",
                    "--random-seed", "11",
                ],
                check=True,
            )

            query_labels = read_query_labels(output_labels)
            counts = {label: query_labels.count((label,)) for label in range(2, 6)}
            self.assertEqual(6, len(query_labels))
            self.assertEqual(1, counts[2])  # maximum candidate: 80%
            self.assertEqual(1, counts[5])  # minimum candidate: 20%
            self.assertEqual(4, counts[3] + counts[4])  # preferred 40%-70% pool


if __name__ == "__main__":
    unittest.main()
