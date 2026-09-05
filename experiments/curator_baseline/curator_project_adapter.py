#!/usr/bin/env python3

import struct
from pathlib import Path

import numpy as np


def load_project_bin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"invalid vector bin header: {path}")
        num_vectors, dim = struct.unpack("<II", header)
        payload = f.read()
    expected_bytes = num_vectors * dim * np.dtype(np.float32).itemsize
    if len(payload) != expected_bytes:
        raise RuntimeError(
            f"invalid vector bin payload: {path}, expected {expected_bytes} bytes, got {len(payload)}"
        )
    return np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False).reshape(num_vectors, dim)


def load_label_sets(path: Path) -> list[frozenset[int]]:
    labels: list[frozenset[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                labels.append(frozenset())
                continue
            labels.append(
                frozenset(int(part.strip()) for part in stripped.split(",") if part.strip())
            )
    return labels



def build_label_inverted_index(base_labels: list[frozenset[int]]) -> dict[int, np.ndarray]:
    postings: dict[int, list[int]] = {}
    for idx, labels in enumerate(base_labels):
        for label in labels:
            postings.setdefault(int(label), []).append(idx)
    return {
        label: np.asarray(ids, dtype=np.uint32)
        for label, ids in postings.items()
    }


def qualified_ids_for_containment_indexed(
    label_index: dict[int, np.ndarray], num_base: int, query_labels: frozenset[int]
) -> np.ndarray:
    if not query_labels:
        return np.arange(num_base, dtype=np.uint32)

    posting_lists = []
    for label in query_labels:
        postings = label_index.get(int(label))
        if postings is None or postings.size == 0:
            return np.array([], dtype=np.uint32)
        posting_lists.append(postings)

    posting_lists.sort(key=lambda values: values.size)
    result = posting_lists[0]
    for postings in posting_lists[1:]:
        result = np.intersect1d(result, postings, assume_unique=True)
        if result.size == 0:
            break
    return np.asarray(result, dtype=np.uint32)

def qualified_ids_for_containment(
    base_labels: list[frozenset[int]], query_labels: frozenset[int]
) -> np.ndarray:
    if not query_labels:
        return np.arange(len(base_labels), dtype=np.uint32)
    return np.asarray(
        [idx for idx, labels in enumerate(base_labels) if query_labels.issubset(labels)],
        dtype=np.uint32,
    )


def load_groundtruth(path: Path, k: int) -> np.ndarray:
    data = path.read_bytes()
    pair_size = struct.calcsize("<If")
    if len(data) % pair_size == 0:
        num_rows = len(data) // (pair_size * k)
        if num_rows * pair_size * k == len(data):
            out = np.empty((num_rows, k), dtype=np.int64)
            offset = 0
            for row in range(num_rows):
                for col in range(k):
                    idx, _dist = struct.unpack_from("<If", data, offset)
                    out[row, col] = -1 if idx == 0xFFFFFFFF else int(idx)
                    offset += pair_size
            return out

    id_size = np.dtype("<u4").itemsize
    if len(data) % (id_size * k) != 0:
        raise RuntimeError(f"invalid groundtruth size for K={k}: {path}")
    out = np.frombuffer(data, dtype="<u4").astype(np.int64).reshape(-1, k)
    out[out == 0xFFFFFFFF] = -1
    return out


def recall_at_k(results: np.ndarray, groundtruth: np.ndarray, k: int) -> float:
    if groundtruth.shape[0] == 0:
        return 0.0
    total = 0.0
    limit = min(k, results.shape[1], groundtruth.shape[1])
    for row in range(groundtruth.shape[0]):
        expected = {int(x) for x in groundtruth[row, :limit] if int(x) >= 0}
        if not expected:
            continue
        actual = {int(x) for x in results[row, :limit] if int(x) >= 0}
        total += len(actual.intersection(expected)) / float(len(expected))
    return total / float(groundtruth.shape[0])
