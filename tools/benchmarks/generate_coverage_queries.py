#!/usr/bin/env python3
import argparse
import csv
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


def write_vecs_bin(path: Path, vectors: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    with path.open("wb") as f:
        f.write(struct.pack("<II", vectors.shape[0], vectors.shape[1]))
        f.write(vectors.tobytes(order="C"))


def read_2d_ints(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            rows.append([int(x) for x in line.split()])
    return rows


def read_group_labels(index_dir: Path):
    rows = read_2d_ints(index_dir / "group_id_to_label_set")
    return [tuple(sorted(row)) for row in rows]


def read_group_to_vec_ids(index_dir: Path):
    path = index_dir / "group_id_to_vec_ids.dat"
    if path.exists():
        return read_2d_ints(path)
    ranges = read_2d_ints(index_dir / "group_id_to_range")
    return [list(range(row[0], row[1])) if len(row) >= 2 else [] for row in ranges]


def build_inverted_index(group_labels):
    label_to_groups = {}
    for gid, labels in enumerate(group_labels):
        if gid == 0 or not labels:
            continue
        for label in labels:
            label_to_groups.setdefault(label, set()).add(gid)
    return label_to_groups


def covered_groups_for(query_labels, label_to_groups):
    groups = None
    for label in query_labels:
        s = label_to_groups.get(label, set())
        groups = set(s) if groups is None else groups & s
        if not groups:
            break
    return sorted(groups or [])


def parse_mix(spec: str):
    items = []
    total = 0.0
    for part in spec.split(","):
        if not part:
            continue
        name, weight = part.split(":")
        w = float(weight)
        if w <= 0:
            continue
        items.append((name, w))
        total += w
    if not items:
        raise ValueError("empty profile mix")
    acc = 0.0
    out = []
    for name, w in items:
        acc += w / total
        out.append((name, acc))
    out[-1] = (out[-1][0], 1.0)
    return out


def choose_profile(rng, mix):
    x = rng.random()
    for name, threshold in mix:
        if x <= threshold:
            return name
    return mix[-1][0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate filtered ANN queries with controlled group/vector coverage."
    )
    parser.add_argument("--base-bin", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--query-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--profile-mix", default="narrow:0.4,medium:0.4,broad:0.2")
    parser.add_argument("--narrow-groups", default="2,8")
    parser.add_argument("--medium-groups", default="9,64")
    parser.add_argument("--broad-groups", default="65,1000000000")
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=2_000_000_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    data = read_vecs_bin(args.base_bin)
    group_labels = read_group_labels(args.index_dir)
    group_to_vec_ids = read_group_to_vec_ids(args.index_dir)
    label_to_groups = build_inverted_index(group_labels)

    ranges_by_profile = {
        "narrow": tuple(int(x) for x in args.narrow_groups.split(",")),
        "medium": tuple(int(x) for x in args.medium_groups.split(",")),
        "broad": tuple(int(x) for x in args.broad_groups.split(",")),
    }
    mix = parse_mix(args.profile_mix)

    parent_groups = [
        gid for gid, labels in enumerate(group_labels)
        if gid > 0 and len(labels) >= args.query_length and gid < len(group_to_vec_ids) and group_to_vec_ids[gid]
    ]
    if not parent_groups:
        raise RuntimeError(
            f"no group has at least {args.query_length} labels. "
            "For singleton-label synthetic data, multi-group subset queries cannot be generated."
        )

    out_vectors = []
    out_labels = []
    out_source_groups = []
    profiles = []

    attempts = 0
    while len(out_vectors) < args.num_queries and attempts < args.max_attempts:
        attempts += 1
        profile = choose_profile(rng, mix)
        min_groups, max_groups = ranges_by_profile[profile]

        parent_gid = rng.choice(parent_groups)
        parent_labels = list(group_labels[parent_gid])
        query_labels = tuple(sorted(rng.sample(parent_labels, args.query_length)))
        groups = covered_groups_for(query_labels, label_to_groups)
        num_groups = len(groups)
        if num_groups < min_groups or num_groups > max_groups:
            continue

        point_count = sum(len(group_to_vec_ids[g]) for g in groups if g < len(group_to_vec_ids))
        if point_count < args.min_points or point_count > args.max_points:
            continue

        source_gid = rng.choice([g for g in groups if g < len(group_to_vec_ids) and group_to_vec_ids[g]])
        source_vid = rng.choice(group_to_vec_ids[source_gid])
        q = np.array(data[source_vid], copy=True)
        if args.noise_std > 0.0:
            q += np_rng.normal(0.0, args.noise_std, size=q.shape).astype(np.float32)

        out_vectors.append(q)
        out_labels.append(query_labels)
        out_source_groups.append(source_gid)
        profiles.append((profile, parent_gid, source_gid, source_vid, num_groups, point_count, query_labels))

    if len(out_vectors) < args.num_queries:
        singleton_groups = sum(1 for gid in parent_groups if len(group_labels[gid]) == 1)
        max_single_label_coverage = 0
        for groups in label_to_groups.values():
            max_single_label_coverage = max(max_single_label_coverage, len(groups))
        raise RuntimeError(
            f"generated only {len(out_vectors)} / {args.num_queries} queries after {attempts} attempts. "
            "Relax group ranges, lower query-length, or use a dataset with richer label overlap. "
            f"diagnostics: parent_groups={len(parent_groups)}, singleton_parent_groups={singleton_groups}, "
            f"max_single_label_matched_groups={max_single_label_coverage}."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.vstack(out_vectors)
    write_vecs_bin(args.out_dir / "query.bin", vectors)

    with (args.out_dir / "query_labels.txt").open("w") as f:
        for labels in out_labels:
            f.write(",".join(str(x) for x in labels) + "\n")

    with (args.out_dir / "query_group_ids.txt").open("w") as f:
        for gid in out_source_groups:
            f.write(f"{gid}\n")

    with (args.out_dir / "query_profile.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "profile", "parent_group_id", "source_group_id", "source_vec_id",
                    "matched_groups", "matched_points", "labels"])
        for qid, row in enumerate(profiles):
            profile, parent_gid, source_gid, source_vid, num_groups, point_count, labels = row
            w.writerow([qid, profile, parent_gid, source_gid, source_vid,
                        num_groups, point_count, " ".join(str(x) for x in labels)])

    counts = {}
    for profile, *_ in profiles:
        counts[profile] = counts.get(profile, 0) + 1
    group_counts = np.array([p[4] for p in profiles], dtype=np.float64)
    point_counts = np.array([p[5] for p in profiles], dtype=np.float64)
    print(f"wrote {len(out_vectors)} queries to {args.out_dir}")
    print(f"profile_counts={counts}")
    print(f"matched_groups avg={group_counts.mean():.2f} p50={np.percentile(group_counts, 50):.1f} p95={np.percentile(group_counts, 95):.1f}")
    print(f"matched_points avg={point_counts.mean():.2f} p50={np.percentile(point_counts, 50):.1f} p95={np.percentile(point_counts, 95):.1f}")


if __name__ == "__main__":
    main()
