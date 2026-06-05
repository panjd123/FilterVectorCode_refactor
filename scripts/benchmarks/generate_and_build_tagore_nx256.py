#!/usr/bin/env python3
import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np


def write_fvecs(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = np.array([data.shape[1]], dtype=np.int32)
    with path.open("wb") as f:
        for row in data.astype(np.float32, copy=False):
            dim.tofile(f)
            row.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--final-degree", type=int, default=64)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    sys.path.insert(0, "/home/graphdb/Tagore/build")
    import Tagore

    out = Path(args.out)
    data_dir = out / "data"
    index_dir = out / "index"
    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    t_all = time.perf_counter()
    with (out / "tagore_nx256_build_times.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gid", "n", "gnn_s_wall", "prune_s_wall", "total_s_wall", "data_path", "index_path"])
        for gid in range(args.groups):
            data_path = data_dir / f"group_{gid}_n{args.nx}.fvecs"
            index_path = index_dir / f"group_{gid}_n{args.nx}.vamana.index"
            if not data_path.exists():
                data = rng.normal(0.0, 1.0, size=(args.nx, args.dim)).astype(np.float32)
                write_fvecs(data_path, data)

            t0 = time.perf_counter()
            ptrs = Tagore.GNN_descent(args.k, args.nx, args.dim, args.iter, str(data_path))
            t1 = time.perf_counter()
            Tagore.Pruning(args.k, args.nx, args.dim, args.final_degree, args.m, ptrs, "Vamana", args.alpha, str(index_path))
            t2 = time.perf_counter()

            row = [gid, args.nx, t1 - t0, t2 - t1, t2 - t0, str(data_path), str(index_path)]
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"{gid + 1}/{args.groups} gid={gid} n={args.nx} total_s={t2 - t0:.6f}", flush=True)
            del ptrs
            gc.collect()

    total_wall = time.perf_counter() - t_all
    totals = [float(r[4]) for r in rows]
    with (out / "tagore_nx256_manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gid", "n", "path", "index_path", "build_ms"])
        for gid, n, _, _, total_s, data_path, index_path in rows:
            writer.writerow([gid, n, data_path, index_path, float(total_s) * 1000.0])
    print(
        "SUMMARY",
        "groups", args.groups,
        "n_total", args.groups * args.nx,
        "wall_total_s", total_wall,
        "sum_group_total_s", sum(totals),
        "mean_ms", (sum(totals) / len(totals)) * 1000.0 if totals else 0.0,
        "max_ms", max(totals) * 1000.0 if totals else 0.0,
        flush=True,
    )


if __name__ == "__main__":
    main()
