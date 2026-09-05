#!/usr/bin/env python3

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curator_project_adapter import (
    build_label_inverted_index,
    load_groundtruth,
    load_label_sets,
    load_project_bin,
    qualified_ids_for_containment,
    qualified_ids_for_containment_indexed,
    recall_at_k,
)


class CuratorProjectAdapterTest(unittest.TestCase):
    def test_load_project_bin_reads_float32_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.bin"
            path.write_bytes(struct.pack("<IIffffff", 3, 2, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0))

            vectors = load_project_bin(path)

            self.assertEqual(vectors.dtype, np.float32)
            np.testing.assert_array_equal(
                vectors,
                np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
            )

    def test_load_label_sets_handles_commas_whitespace_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            path.write_text("1, 2,3\n\n7\n", encoding="utf-8")

            labels = load_label_sets(path)

            self.assertEqual(labels, [frozenset({1, 2, 3}), frozenset(), frozenset({7})])

    def test_qualified_ids_for_containment_returns_superset_rows(self):
        base_labels = [
            frozenset({1, 2, 3}),
            frozenset({1}),
            frozenset({2, 3}),
            frozenset({1, 2}),
        ]

        ids = qualified_ids_for_containment(base_labels, frozenset({1, 2}))

        np.testing.assert_array_equal(ids, np.array([0, 3], dtype=np.uint32))

    def test_indexed_containment_matches_scan_and_handles_empty_or_missing_labels(self):
        base_labels = [
            frozenset({1, 2}),
            frozenset({2}),
            frozenset({1, 2, 3}),
            frozenset({3}),
            frozenset(),
        ]
        label_index = build_label_inverted_index(base_labels)

        np.testing.assert_array_equal(
            qualified_ids_for_containment_indexed(label_index, len(base_labels), frozenset({1, 2})),
            qualified_ids_for_containment(base_labels, frozenset({1, 2})),
        )
        np.testing.assert_array_equal(
            qualified_ids_for_containment_indexed(label_index, len(base_labels), frozenset()),
            np.arange(len(base_labels), dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            qualified_ids_for_containment_indexed(label_index, len(base_labels), frozenset({99})),
            np.array([], dtype=np.uint32),
        )

    def test_load_groundtruth_reads_project_uint_float_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.bin"
            path.write_bytes(
                struct.pack(
                    "<IfIfIfIf",
                    3,
                    0.1,
                    5,
                    0.2,
                    7,
                    0.3,
                    11,
                    0.4,
                )
            )

            groundtruth = load_groundtruth(path, 2)

            np.testing.assert_array_equal(
                groundtruth,
                np.array([[3, 5], [7, 11]], dtype=np.int64),
            )

    def test_recall_at_k_ignores_padded_minus_one_results(self):
        results = np.array([[3, -1], [2, 9]], dtype=np.int64)
        groundtruth = np.array([[3, 5], [7, 9]], dtype=np.int64)

        recall = recall_at_k(results, groundtruth, 2)

        self.assertAlmostEqual(recall, 0.5)


if __name__ == "__main__":
    unittest.main()
