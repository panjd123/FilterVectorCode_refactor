#!/usr/bin/env python3
import argparse
import collections
import json
import math
import random
import re
from pathlib import Path


def parse_label_line(line):
    return [int(value) for value in re.findall(r"\d+", line)]


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def parse_label_set(value):
    if not value:
        return set()
    return {int(item) for item in re.findall(r"\d+", value)}


def read_labels(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rows.append(parse_label_line(line))
    return rows


def label_frequency(rows):
    freq = collections.Counter()
    for labels in rows:
        freq.update(labels)
    return freq


def summarize_rows(rows):
    freq = label_frequency(rows)
    labelset_counts = collections.Counter(tuple(row) for row in rows)
    lengths = sorted(len(row) for row in rows)
    counts = sorted(freq.values(), reverse=True)
    total_assignments = sum(counts)
    top = freq.most_common(20)
    return {
        "num_points": len(rows),
        "total_assignments": total_assignments,
        "unique_labels": len(freq),
        "unique_labelsets": len(labelset_counts),
        "duplicate_labelsets": sum(1 for value in labelset_counts.values() if value > 1),
        "labels_per_point": {
            "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "p50": percentile(lengths, 50),
            "p90": percentile(lengths, 90),
            "p95": percentile(lengths, 95),
            "p99": percentile(lengths, 99),
            "max": max(lengths) if lengths else 0,
        },
        "max_labels_per_point": max(lengths) if lengths else 0,
        "label_freq": {
            "min": min(counts) if counts else 0,
            "p50": percentile(sorted(counts), 50),
            "p90": percentile(sorted(counts), 90),
            "p95": percentile(sorted(counts), 95),
            "p99": percentile(sorted(counts), 99),
            "max": max(counts) if counts else 0,
            "count_eq_1": sum(1 for value in counts if value == 1),
            "count_le_10": sum(1 for value in counts if value <= 10),
            "count_le_100": sum(1 for value in counts if value <= 100),
        },
        "top_label_shares": {
            "top1": (sum(counts[:1]) / total_assignments) if total_assignments else 0.0,
            "top5": (sum(counts[:5]) / total_assignments) if total_assignments else 0.0,
            "top10": (sum(counts[:10]) / total_assignments) if total_assignments else 0.0,
            "top100": (sum(counts[:100]) / total_assignments) if total_assignments else 0.0,
        },
        "top20_labels": [{"label": label, "count": count} for label, count in top],
    }


def build_row_mix_rows(zipf_rows, old_rows, args):
    num_rows = len(zipf_rows)
    old_target = int(round(num_rows * args.old_row_ratio))
    split = old_target

    if args.old_row_placement == "first":
        old_range = (0, split)
        zipf_range = (split, num_rows)
        output_rows = [list(old_rows[idx]) if idx < split else list(zipf_rows[idx]) for idx in range(num_rows)]
    else:
        zipf_range = (0, num_rows - old_target)
        old_range = (num_rows - old_target, num_rows)
        output_rows = [
            list(zipf_rows[idx]) if idx < num_rows - old_target else list(old_rows[idx])
            for idx in range(num_rows)
        ]

    return output_rows, {
        "old_rows": old_target,
        "zipf_rows": num_rows - old_target,
        "old_row_ratio_actual": (old_target / num_rows) if num_rows else 0.0,
        "old_range": list(old_range),
        "zipf_range": list(zipf_range),
    }


def postprocess_rows(rows, args, mix_stats=None):
    processed = [list(row) for row in rows]
    stats = {
        "filled_empty_rows": 0,
        "label_id_mapping_size": 0,
        "label_id_min": 0,
        "label_id_max": 0,
        "compact_label_ids": bool(args.compact_label_ids),
    }

    if args.fill_empty_with_previous:
        previous = None
        for idx, row in enumerate(processed):
            if row:
                previous = list(row)
                continue
            if previous is None:
                raise ValueError("cannot fill empty first row because no previous row exists")
            processed[idx] = list(previous)
            stats["filled_empty_rows"] += 1

    if args.compact_label_ids:
        if args.mode == "row-mix" and mix_stats and "old_range" in mix_stats and "zipf_range" in mix_stats:
            old_start, old_end = mix_stats["old_range"]
            zipf_start, zipf_end = mix_stats["zipf_range"]
            old_labels = sorted({label for row in processed[old_start:old_end] for label in row})
            zipf_labels = sorted({label for row in processed[zipf_start:zipf_end] for label in row})
            old_mapping = {label: idx + 1 for idx, label in enumerate(old_labels)}
            zipf_offset = len(old_mapping)
            zipf_mapping = {label: zipf_offset + idx + 1 for idx, label in enumerate(zipf_labels)}
            for idx in range(old_start, old_end):
                processed[idx] = [old_mapping[label] for label in processed[idx]]
            for idx in range(zipf_start, zipf_end):
                processed[idx] = [zipf_mapping[label] for label in processed[idx]]
            stats["label_id_mapping_size"] = len(old_mapping) + len(zipf_mapping)
            stats["label_id_old_mapping_size"] = len(old_mapping)
            stats["label_id_zipf_mapping_size"] = len(zipf_mapping)
        else:
            labels = sorted({label for row in processed for label in row})
            mapping = {label: idx + 1 for idx, label in enumerate(labels)}
            processed = [[mapping[label] for label in row] for row in processed]
            stats["label_id_mapping_size"] = len(mapping)
        stats["label_id_min"] = 1 if stats["label_id_mapping_size"] else 0
        stats["label_id_max"] = stats["label_id_mapping_size"]

    return processed, stats


def build_hybrid_rows(zipf_rows, old_rows, args):
    old_freq = label_frequency(old_rows)
    exclude_labels = parse_label_set(args.exclude_labels)
    rng = random.Random(args.seed)
    output_rows = []
    dropped_zipf_labels = 0
    sampled_tail_labels = 0
    eligible_tail_labels = 0

    for zipf_labels, old_labels in zip(zipf_rows, old_rows):
        core = sorted(set(zipf_labels))
        if len(core) > args.max_labels:
            dropped_zipf_labels += len(core) - args.max_labels
            core = core[: args.max_labels]

        core_set = set(core)
        tail_pool = []
        for label in old_labels:
            if label in core_set or label in exclude_labels:
                continue
            count = old_freq[label]
            if count < args.tail_min_freq:
                continue
            if args.tail_max_freq > 0 and count > args.tail_max_freq:
                continue
            tail_pool.append(label)

        tail_pool = sorted(set(tail_pool))
        eligible_tail_labels += len(tail_pool)
        rng.shuffle(tail_pool)
        selected_tail = []
        for label in tail_pool:
            if len(selected_tail) >= args.tail_max_per_point:
                break
            if rng.random() <= args.tail_prob:
                selected_tail.append(label)

        budget = max(0, args.max_labels - len(core))
        selected_tail = selected_tail[:budget]
        sampled_tail_labels += len(selected_tail)
        output_rows.append(sorted(set(core).union(selected_tail)))

    return output_rows, {
        "eligible_tail_labels": eligible_tail_labels,
        "sampled_tail_labels": sampled_tail_labels,
        "dropped_zipf_labels": dropped_zipf_labels,
    }


def write_rows(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(str(label) for label in row))
            f.write("\n")


def generate(args):
    zipf_rows = read_labels(args.zipf_base)
    old_rows = read_labels(args.old_base)
    if len(zipf_rows) != len(old_rows):
        raise ValueError(
            f"zipf-base and old-base must have the same number of rows: "
            f"{len(zipf_rows)} != {len(old_rows)}"
        )

    if args.mode == "row-mix":
        output_rows, mix_stats = build_row_mix_rows(zipf_rows, old_rows, args)
    else:
        output_rows, mix_stats = build_hybrid_rows(zipf_rows, old_rows, args)
    output_rows, postprocess_stats = postprocess_rows(output_rows, args, mix_stats)
    write_rows(args.output, output_rows)

    summary = {
        "zipf_base": str(args.zipf_base),
        "old_base": str(args.old_base),
        "output": str(args.output),
        "num_points": len(output_rows),
        "seed": args.seed,
        "parameters": {
            "mode": args.mode,
            "old_row_ratio": args.old_row_ratio,
            "old_row_placement": args.old_row_placement,
            "no_manifest": args.no_manifest,
            "fill_empty_with_previous": args.fill_empty_with_previous,
            "compact_label_ids": args.compact_label_ids,
            "tail_prob": args.tail_prob,
            "tail_min_freq": args.tail_min_freq,
            "tail_max_freq": args.tail_max_freq,
            "tail_max_per_point": args.tail_max_per_point,
            "max_labels": args.max_labels,
            "exclude_labels": sorted(parse_label_set(args.exclude_labels)),
        },
        "mix_stats": mix_stats,
        "postprocess_stats": postprocess_stats,
        "zipf_stats": summarize_rows(zipf_rows),
        "old_stats": summarize_rows(old_rows),
        "output_stats": summarize_rows(output_rows),
    }

    if not args.no_manifest:
        manifest = Path(args.manifest) if args.manifest else Path(str(args.output) + ".manifest.json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a Zipf-core + old-tail hybrid base label file for UNG/ELS stress sweeps."
    )
    parser.add_argument("--zipf-base", type=Path, required=True, help="Zipf-style base label file.")
    parser.add_argument("--old-base", type=Path, required=True, help="Old complex base label file.")
    parser.add_argument("--output", type=Path, required=True, help="Output hybrid base label file.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON summary path. Defaults to OUTPUT.manifest.json.")
    parser.add_argument("--no-manifest", action="store_true", help="Do not write a manifest JSON file.")
    parser.add_argument("--fill-empty-with-previous", action="store_true", help="Replace empty output rows with the previous non-empty row.")
    parser.add_argument("--compact-label-ids", action="store_true", help="Remap all output labels in ascending original-id order to contiguous ids 1..N.")
    parser.add_argument("--mode", choices=("tail-inject", "row-mix"), default="tail-inject", help="Hybrid strategy.")
    parser.add_argument("--old-row-ratio", type=float, default=0.5, help="For row-mix mode, fraction of contiguous rows copied from old-base.")
    parser.add_argument("--old-row-placement", choices=("first", "last"), default="first", help="For row-mix mode, place the old-base block at the first or last rows.")
    parser.add_argument("--seed", type=int, default=20260722, help="Random seed for reproducible tail sampling.")
    parser.add_argument("--tail-prob", type=float, default=0.45, help="Probability of keeping each eligible old-tail label.")
    parser.add_argument("--tail-min-freq", type=int, default=2, help="Minimum old-label frequency eligible for tail injection.")
    parser.add_argument(
        "--tail-max-freq",
        type=int,
        default=10000,
        help="Maximum old-label frequency eligible for tail injection. Use 0 for no upper bound.",
    )
    parser.add_argument("--tail-max-per-point", type=int, default=4, help="Maximum sampled old-tail labels per point.")
    parser.add_argument("--max-labels", type=int, default=32, help="Hard cap on output labels per point.")
    parser.add_argument(
        "--exclude-labels",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Old labels never injected as tail. Comma/space separated.",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.old_row_ratio <= 1.0:
        parser.error("--old-row-ratio must be in [0, 1]")
    if not 0.0 <= args.tail_prob <= 1.0:
        parser.error("--tail-prob must be in [0, 1]")
    if args.tail_min_freq < 1:
        parser.error("--tail-min-freq must be >= 1")
    if args.tail_max_freq < 0:
        parser.error("--tail-max-freq must be >= 0")
    if args.tail_max_per_point < 0:
        parser.error("--tail-max-per-point must be >= 0")
    if args.max_labels < 1:
        parser.error("--max-labels must be >= 1")
    return args


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
