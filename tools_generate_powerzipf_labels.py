#!/usr/bin/env python3
import argparse
import collections
import json
import math
import random
from pathlib import Path


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--num-points', type=int, default=1_000_000)
    ap.add_argument('--num-labels', type=int, default=30)
    ap.add_argument('--alpha', type=float, default=0.85)
    ap.add_argument('--c', type=float, default=0.75)
    ap.add_argument('--pmax', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=20260519)
    ap.add_argument('--manifest', default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probs = [min(args.pmax, args.c / ((i + 1) ** args.alpha)) for i in range(args.num_labels)]
    counts = collections.Counter()
    card_counts = collections.Counter()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for _ in range(args.num_points):
            labels = tuple(i + 1 for i, p in enumerate(probs) if rng.random() < p)
            if not labels:
                labels = (rng.randrange(1, args.num_labels + 1),)
            counts[labels] += 1
            card_counts[len(labels)] += 1
            f.write(','.join(map(str, labels)))
            f.write('\n')

    group_sizes = sorted(counts.values())
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    summary = {
        'num_points': args.num_points,
        'num_labels': args.num_labels,
        'alpha': args.alpha,
        'c': args.c,
        'pmax': args.pmax,
        'seed': args.seed,
        'probabilities': probs,
        'expected_label_cardinality': sum(probs),
        'num_groups': len(counts),
        'singleton_groups': sum(1 for v in group_sizes if v == 1),
        'group_size': {
            'p50': percentile(group_sizes, 50),
            'p90': percentile(group_sizes, 90),
            'p95': percentile(group_sizes, 95),
            'p99': percentile(group_sizes, 99),
            'max': max(group_sizes),
        },
        'top1_share': top[0][1] / args.num_points,
        'top10_share': sum(v for _, v in top[:10]) / args.num_points,
        'label_cardinality_hist': dict(sorted(card_counts.items())),
        'top20_groups': [{'labels': list(k), 'size': v} for k, v in top],
    }
    manifest = Path(args.manifest) if args.manifest else out.with_suffix(out.suffix + '.manifest.json')
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
