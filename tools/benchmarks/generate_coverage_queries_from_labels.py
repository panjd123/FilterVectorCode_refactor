#!/usr/bin/env python3
import argparse
import csv
import random
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_header(path: Path):
    with path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid vector file header: {path}")
    return struct.unpack("<II", header)


def read_one_vector(path: Path, dim: int, vid: int):
    with path.open("rb") as f:
        f.seek(8 + vid * dim * 4)
        raw = f.read(dim * 4)
    if len(raw) != dim * 4:
        raise ValueError(f"failed to read vector id {vid} from {path}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_vecs_bin(path: Path, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    with path.open("wb") as f:
        f.write(struct.pack("<II", arr.shape[0], arr.shape[1]))
        f.write(arr.tobytes(order="C"))


def parse_labels(line: str):
    return tuple(sorted(int(x) for x in line.strip().split(",") if x))


def read_groups_from_labels(label_file: Path, expected_n: int):
    group_to_vec_ids = []
    group_labels = [tuple()]
    label_to_gid = {}

    with label_file.open() as f:
        for vid, line in enumerate(f):
            labels = parse_labels(line)
            gid = label_to_gid.get(labels)
            if gid is None:
                gid = len(group_labels)
                label_to_gid[labels] = gid
                group_labels.append(labels)
                group_to_vec_ids.append([])
            group_to_vec_ids[gid - 1].append(vid)

    if vid + 1 != expected_n:
        raise ValueError(f"label count mismatch: labels={vid + 1}, vectors={expected_n}")
    return group_labels, [[]] + group_to_vec_ids


def build_inverted_index(group_labels):
    label_to_groups = defaultdict(set)
    for gid, labels in enumerate(group_labels):
        if gid == 0:
            continue
        for label in labels:
            label_to_groups[label].add(gid)
    return label_to_groups


def covered_groups_for(query_labels, label_to_groups):
    groups = None
    for label in query_labels:
        current = label_to_groups.get(label, set())
        groups = set(current) if groups is None else groups & current
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
        weight = float(weight)
        if weight <= 0:
            continue
        items.append((name, weight))
        total += weight
    if not items:
        raise ValueError("empty profile mix")
    acc = 0.0
    out = []
    for name, weight in items:
        acc += weight / total
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
        description="Generate coverage-style filtered ANN queries directly from a label file."
    )
    parser.add_argument("--base-bin", required=True, type=Path)
    parser.add_argument("--label-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--query-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=300000)
    parser.add_argument("--profile-mix", default="narrow:0.4,medium:0.4,broad:0.2")
    parser.add_argument("--narrow-groups", default="2,8")
    parser.add_argument("--medium-groups", default="9,64")
    parser.add_argument("--broad-groups", default="65,1000000000")
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=2_000_000_000)
    args = parser.parse_args()

    n, dim = read_header(args.base_bin)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    group_labels, group_to_vec_ids = read_groups_from_labels(args.label_file, n)
    label_to_groups = build_inverted_index(group_labels)

    ranges_by_profile = {
        "narrow": tuple(int(x) for x in args.narrow_groups.split(",")),
        "medium": tuple(int(x) for x in args.medium_groups.split(",")),
        "broad": tuple(int(x) for x in args.broad_groups.split(",")),
    }
    mix = parse_mix(args.profile_mix)

    parent_groups = [
        gid for gid, labels in enumerate(group_labels)
        if gid > 0 and len(labels) >= args.query_length and group_to_vec_ids[gid]
    ]
    if not parent_groups:
        raise RuntimeError(f"no group has at least {args.query_length} labels")

    vectors = []
    out_labels = []
    out_source_groups = []
    profiles = []

    attempts = 0
    while len(vectors) < args.num_queries and attempts < args.max_attempts:
        attempts += 1
        profile = choose_profile(rng, mix)
        min_groups, max_groups = ranges_by_profile[profile]
        parent_gid = rng.choice(parent_groups)
        query_labels = tuple(sorted(rng.sample(group_labels[parent_gid], args.query_length)))
        groups = covered_groups_for(query_labels, label_to_groups)
        if not (min_groups <= len(groups) <= max_groups):
            continue
        point_count = sum(len(group_to_vec_ids[gid]) for gid in groups)
        if point_count < args.min_points or point_count > args.max_points:
            continue

        source_gid = rng.choice([gid for gid in groups if group_to_vec_ids[gid]])
        source_vid = rng.choice(group_to_vec_ids[source_gid])
        q = read_one_vector(args.base_bin, dim, source_vid)
        if args.noise_std > 0.0:
            q += np_rng.normal(0.0, args.noise_std, size=q.shape).astype(np.float32)

        vectors.append(q)
        out_labels.append(query_labels)
        out_source_groups.append(source_gid)
        profiles.append((profile, parent_gid, source_gid, source_vid, len(groups), point_count, query_labels))

    if len(vectors) < args.num_queries:
        max_single_label_coverage = max((len(groups) for groups in label_to_groups.values()), default=0)
        raise RuntimeError(
            f"generated only {len(vectors)} / {args.num_queries} after {attempts} attempts; "
            f"parent_groups={len(parent_groups)}, max_single_label_matched_groups={max_single_label_coverage}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_vecs_bin(args.out_dir / "query.bin", vectors)
    with (args.out_dir / "query_labels.txt").open("w") as f:
        for labels in out_labels:
            f.write(",".join(str(x) for x in labels) + "\n")
    with (args.out_dir / "query_group_ids.txt").open("w") as f:
        for gid in out_source_groups:
            f.write(f"{gid}\n")
    with (args.out_dir / "query_profile.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "profile", "parent_group_id", "source_group_id", "source_vec_id",
                         "matched_groups", "matched_points", "labels"])
        for qid, row in enumerate(profiles):
            profile, parent_gid, source_gid, source_vid, matched_groups, matched_points, labels = row
            writer.writerow([qid, profile, parent_gid, source_gid, source_vid,
                             matched_groups, matched_points, " ".join(str(x) for x in labels)])

    counts = defaultdict(int)
    for profile, *_ in profiles:
        counts[profile] += 1
    group_counts = np.array([p[4] for p in profiles], dtype=np.float64)
    point_counts = np.array([p[5] for p in profiles], dtype=np.float64)
    print(f"wrote {len(vectors)} queries to {args.out_dir}")
    print(f"profile_counts={dict(counts)}")
    print(f"matched_groups avg={group_counts.mean():.2f} p50={np.percentile(group_counts, 50):.1f} p95={np.percentile(group_counts, 95):.1f}")
    print(f"matched_points avg={point_counts.mean():.2f} p50={np.percentile(point_counts, 50):.1f} p95={np.percentile(point_counts, 95):.1f}")


if __name__ == "__main__":
    main()
