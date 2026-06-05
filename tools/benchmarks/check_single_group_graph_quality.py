#!/usr/bin/env python3
import argparse
import heapq
import random
import struct
from pathlib import Path

import numpy as np


def read_vecs_bin(path: Path):
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"invalid vector file header: {path}")
        n, dim = struct.unpack("<II", header)
        data = np.fromfile(f, dtype=np.float32, count=n * dim)
    if data.size != n * dim:
        raise ValueError(f"vector file is truncated: expected {n * dim}, got {data.size}")
    return data.reshape(n, dim)


def read_group_ranges(index_dir: Path):
    ranges = []
    with (index_dir / "group_id_to_range").open() as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                ranges.append((int(parts[0]), int(parts[1])))
    return ranges


def read_graph_lines(index_dir: Path, start: int, end: int):
    graph = {}
    with (index_dir / "graph").open() as f:
        for line_no, line in enumerate(f):
            if line_no >= end:
                break
            if line_no < start:
                continue
            parts = [int(x) for x in line.split()]
            if not parts:
                continue
            graph[parts[0]] = parts[1:]
    return graph


def exact_topk_ids(vectors: np.ndarray, local_id: int, k: int):
    q = vectors[local_id]
    diff = vectors - q
    dists = np.einsum("ij,ij->i", diff, diff)
    dists[local_id] = np.inf
    if k >= len(dists) - 1:
        return set(np.argsort(dists)[:k].tolist())
    idx = np.argpartition(dists, k)[:k]
    return set(idx[np.argsort(dists[idx])].tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Check one group's stored graph neighbors against exact in-group L2 topK."
    )
    parser.add_argument("--base-bin", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--group-id", required=True, type=int)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--details", type=int, default=5)
    args = parser.parse_args()

    ranges = read_group_ranges(args.index_dir)
    if args.group_id < 1 or args.group_id >= len(ranges):
        raise ValueError(f"group id {args.group_id} is out of range [1, {len(ranges) - 1}]")
    start, end = ranges[args.group_id]
    if end <= start:
        raise ValueError(f"group {args.group_id} is empty")

    data = read_vecs_bin(args.base_bin)
    group_vectors = data[start:end]
    graph = read_graph_lines(args.index_dir, start, end)

    rng = random.Random(args.seed)
    local_ids = list(range(end - start))
    if args.sample > 0 and args.sample < len(local_ids):
        local_ids = sorted(rng.sample(local_ids, args.sample))

    recalls = []
    degrees = []
    details = []
    for local_id in local_ids:
        global_id = start + local_id
        approx_global = graph.get(global_id, [])
        approx_local = [x - start for x in approx_global if start <= x < end and x != global_id]
        approx_set = set(approx_local[: args.k])
        exact = exact_topk_ids(group_vectors, local_id, args.k)
        hit = len(approx_set & exact)
        recall = hit / float(args.k)
        recalls.append(recall)
        degrees.append(len(approx_local))
        if len(details) < args.details:
            details.append((global_id, recall, hit, len(approx_local), sorted(list(exact))[: args.k], approx_local[: args.k]))

    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    p50 = float(np.percentile(recalls, 50)) if recalls else 0.0
    p95 = float(np.percentile(recalls, 95)) if recalls else 0.0
    avg_degree = float(np.mean(degrees)) if degrees else 0.0

    print(f"group_id={args.group_id} range=[{start},{end}) points={end - start}")
    print(f"sampled={len(local_ids)} k={args.k}")
    print(f"avg_recall@{args.k}={avg_recall:.6f} p50={p50:.6f} p95={p95:.6f} avg_in_group_degree={avg_degree:.3f}")
    for global_id, recall, hit, degree, exact, approx in details:
        print(f"detail global_id={global_id} recall={recall:.3f} hit={hit}/{args.k} degree={degree}")
        print(f"  exact_local={exact}")
        print(f"  graph_local={approx}")


if __name__ == "__main__":
    main()
