#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curator_backend import CuratorBackend, import_curator_class


class FakeCurator:
    instances = []

    def __init__(self, d, nlist, **kwargs):
        self.d = d
        self.nlist = nlist
        self.kwargs = kwargs
        self.trained = None
        self.created = []
        self.search_calls = []
        FakeCurator.instances.append(self)

    def train(self, vectors):
        self.trained = vectors.copy()

    def create(self, vector, label):
        self.created.append((vector.copy(), label))

    def search_with_bitmap_filter(self, query, k, qualified_labels):
        self.search_calls.append((query.copy(), k, qualified_labels.copy()))
        return (
            np.array([[0.1, 0.2]], dtype=np.float32),
            np.array([[int(qualified_labels[0]), -1]], dtype=np.int64),
        )

    def get_last_search_profile(self):
        return {
            "search_time_ms": 0.25,
            "qualified_labels_count": 1,
            "visited_points": 7,
            "visited_edges": 9,
            "distance_computations": 11,
        }


class FakeInnerCuratorIndex:
    def __init__(self):
        self.mapping_calls = []
        self.optimized_calls = []

    def get_label_to_vid_mapping(self, labels):
        self.mapping_calls.append(labels.copy())
        return np.array([30, 10, 20], dtype=np.uint64)[: labels.shape[0]]

    def search_with_bitmap_filter_optimized(self, query_batch, k, sorted_vids):
        self.optimized_calls.append((query_batch.copy(), k, sorted_vids.copy()))
        return (
            np.array([[0.1, 0.2]], dtype=np.float32),
            np.array([[42, -1]], dtype=np.int64),
        )


class OptimizedFakeCurator(FakeCurator):
    def __init__(self, d, nlist, **kwargs):
        super().__init__(d, nlist, **kwargs)
        self.index = FakeInnerCuratorIndex()


class CuratorBackendTest(unittest.TestCase):
    def setUp(self):
        FakeCurator.instances.clear()

    def test_build_trains_and_inserts_project_ids(self):
        backend = CuratorBackend(FakeCurator, {"d": 2, "nlist": 4, "nprobe": 2})
        vectors = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        ids = np.array([10, 11], dtype=np.uint32)

        backend.build(vectors, ids)

        fake = FakeCurator.instances[0]
        np.testing.assert_array_equal(fake.trained, vectors)
        self.assertEqual([label for _vec, label in fake.created], [10, 11])

    def test_search_uses_uint32_qualified_ids_and_pads_results(self):
        backend = CuratorBackend(FakeCurator, {"d": 2, "nlist": 4})
        backend.build(np.array([[0.0, 0.0]], dtype=np.float32), np.array([8], dtype=np.uint32))

        ids, profile = backend.search(
            np.array([0.0, 0.0], dtype=np.float32),
            2,
            np.array([8], dtype=np.uint32),
        )

        fake = FakeCurator.instances[0]
        self.assertEqual(fake.search_calls[0][1], 2)
        self.assertEqual(fake.search_calls[0][2].dtype, np.uint32)
        np.testing.assert_array_equal(ids, np.array([8, -1], dtype=np.int64))
        self.assertEqual(profile["qualified_labels_count"], 1)
        self.assertEqual(profile["visited_points"], 7)
        self.assertEqual(profile["visited_edges"], 9)
        self.assertEqual(profile["distance_computations"], 11)

    def test_prepared_filter_uses_sorted_internal_vids_for_optimized_search(self):
        backend = CuratorBackend(OptimizedFakeCurator, {"d": 2, "nlist": 4})

        prepared = backend.prepare_filter(np.array([5, 6, 7], dtype=np.uint32))
        ids, profile = backend.search(
            np.array([0.0, 0.0], dtype=np.float32),
            2,
            prepared,
        )

        fake = FakeCurator.instances[0]
        np.testing.assert_array_equal(fake.index.mapping_calls[0], np.array([5, 6, 7], dtype=np.uint32))
        np.testing.assert_array_equal(prepared["sorted_vids"], np.array([10, 20, 30], dtype=np.uint64))
        np.testing.assert_array_equal(fake.index.optimized_calls[0][2], np.array([10, 20, 30], dtype=np.uint64))
        np.testing.assert_array_equal(ids, np.array([42, -1], dtype=np.int64))
        self.assertEqual(profile["qualified_labels_count"], 3)

    def test_import_curator_class_mentions_curator_repo_when_unavailable(self):
        with mock.patch.dict("sys.modules", {"indexes.curator": None}):
            with self.assertRaisesRegex(ImportError, "curator_repo"):
                import_curator_class(None)


if __name__ == "__main__":
    unittest.main()
