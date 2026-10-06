#!/usr/bin/env python3
"""Reproduce descriptive timing ranges from the published warm-repeat rows.

These are the original two repeats at each frozen Recall crossing. The four
cross-repeat ratios are an observed range, not a confidence interval, and do
not assume that executions of different configurations were paired.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from summarize_base_topology_factorial import WORKLOAD_ORDER
from plot_topology_factorial import CODES


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def positive(row: dict, name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'invalid positive {name}: {value}')
    return value


def validate_points(points: list[dict], cells: list[dict]) -> dict:
    expected = {}
    for row in cells:
        key = (row['topology_code'], row['workload'])
        if key in expected:
            raise ValueError(f'duplicate factorial cell: {key}')
        expected[key] = row
    full_grid = {(code, workload) for code in CODES for workload in WORKLOAD_ORDER}
    if set(expected) != full_grid:
        raise ValueError('the frozen factorial must contain the complete 14 by 9 grid')
    measured = {}
    for row in points:
        key = (row['topology_code'], row['workload'])
        if key in measured:
            raise ValueError(f'duplicate warm-repeat cell: {key}')
        cell = expected.get(key)
        if cell is None or cell['status'] != 'complete':
            raise ValueError(f'no frozen crossing for warm-repeat cell: {key}')
        if row['method'] != cell['method'] or int(row['lsearch']) != int(cell['lsearch']):
            raise ValueError(f'wrong method/capacity: {key}')
        times = [positive(row, f'warm_{i}_ms') for i in (1, 2)]
        recalls = [float(row[f'warm_{i}_recall']) for i in (1, 2)]
        if not all(math.isfinite(r) and .9 <= r <= 1 for r in recalls):
            raise ValueError(f'non-crossing warm Recall: {key}')
        # Every frozen Amazon batch contains exactly 1,000 queries.
        qps = 1_000_000 / statistics.median(times)
        cv = statistics.stdev(times) / statistics.mean(times)
        checks = ((qps, float(cell['warm_median_qps'])),
                  (qps, float(row['qps_at_median_time'])),
                  (cv, float(row['warm_cv'])),
                  (min(recalls), float(cell['recall_min'])),
                  (float(row['mean_selectivity']), float(cell['mean_selectivity'])))
        if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in checks):
            raise ValueError(f'repeat values do not reproduce frozen summary: {key}')
        measured[key] = {'times': times, 'qps': qps, 'cv': cv,
                         'capacity': int(cell['lsearch']),
                         'mean_selectivity': float(cell['mean_selectivity'])}
    crossings = {k for k, r in expected.items() if r['status'] == 'complete'}
    if set(measured) != crossings:
        raise ValueError(f'missing measured crossings: {sorted(crossings - set(measured))}')
    return measured


def observed_ratio(left: dict, right: dict) -> tuple[float, float]:
    # QPS_left/QPS_right = time_right/time_left for equal batch sizes.
    ratios = [b/a for a in left['times'] for b in right['times']]
    return min(ratios), max(ratios)


def summarize(points: Path, factorial: Path, output: Path) -> dict:
    measured = validate_points(read_csv(points), read_csv(factorial))
    comparisons = []
    for a, b in [('T', 'L'), ('TLT', 'L'), ('TLT', 'TL'), ('TLT', 'TLL')]:
        for workload in WORKLOAD_ORDER:
            left, right = measured.get((a, workload)), measured.get((b, workload))
            if left is None or right is None:
                continue
            lower, upper = observed_ratio(left, right)
            comparisons.append({'comparison': a+'/'+b, 'workload': workload,
                                'median_time_qps_ratio': left['qps']/right['qps'],
                                'min_cross_repeat_ratio': lower,
                                'max_cross_repeat_ratio': upper,
                                'observed_ranges_overlap': lower <= 1 <= upper})
    latex = [
        r'\newcommand{\warmRepeatRanges}{%',
        r'\begin{table}[H]', r'\centering\small',
        r'\caption{Selected capacities and observed QPS-ratio ranges from the two original warm repeats. '
        r'The intervals span all four cross-repeat ratios, not confidence bounds. '
        r'TL, TLT, and TLL denote \texttt{1L[T|L]}, \texttt{2L[T|LT]}, and \texttt{2L[T|LL]}.}',
        r'\label{tab:warm-repeat-ranges}',
        r'\begin{tabular}{r rrr cc}', r'\toprule',
        r'Mean sel. & Plain $L_s$ & TL $L_s$ & TLT $L_s$ & TLT/TL range & TLT/TLL range \\',
        r'\midrule',
    ]
    for workload in WORKLOAD_ORDER:
        base, one, two, alt = [measured[c, workload] for c in ('L', 'TL', 'TLT', 'TLL')]
        depth = observed_ratio(two, one)
        topology = observed_ratio(two, alt)
        latex.append(f'{100*base["mean_selectivity"]:.3f}\\% & '
                     f'{base["capacity"]:,} & {one["capacity"]:,} & {two["capacity"]:,} & '
                     f'[{depth[0]:.3f}, {depth[1]:.3f}] & '
                     f'[{topology[0]:.3f}, {topology[1]:.3f}]'+r' \\')
    latex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '}']
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output/'warm_repeat_comparisons.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    tex_path = output/'generated_warm_repeat_ranges.tex'
    tex_path.write_text('\n'.join(latex)+'\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        'scope': 'Descriptive reanalysis of original warm timings; no new experiment.',
        'warm_repeats_per_point': 2, 'queries_per_batch': 1000,
        'crossings': len(measured), 'comparison_rows': len(comparisons),
        'cv_median': statistics.median(r['cv'] for r in measured.values()),
        'cv_max': max(r['cv'] for r in measured.values()),
        'not_a_confidence_interval': True,
        'range_definition': 'min(time_B)/max(time_A) through max(time_B)/min(time_A).',
        'limitation': 'Observed overlap or separation does not establish statistical significance or reproducibility across independent runs.',
        'inputs': [{'name': p.name, 'sha256': sha(p)} for p in (points, factorial)],
        'script_sha256': sha(Path(__file__)),
        'outputs': [{'name': p.name, 'sha256': sha(p)} for p in (csv_path, tex_path)],
    }
    (output/'summary_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('points', type=Path)
    parser.add_argument('factorial', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    result = summarize(args.points, args.factorial, args.output_dir)
    print(f'Validated {result["crossings"]} crossings; wrote {result["comparison_rows"]} observed ranges.')


if __name__ == '__main__':
    main()
