#!/usr/bin/env python3
"""Plot the complete, frozen topology table without rerunning experiments.

Both panels use the same conservative Recall crossings. Missing artifacts
are errors; a measured no-crossing cell remains NC, never zero throughput.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from summarize_base_topology_factorial import WORKLOAD_ORDER, display_code

CODES = ('L', 'LL', 'LT', 'LLL', 'LLT', 'LTL', 'LTT',
         'T', 'TL', 'TT', 'TLL', 'TLT', 'TTL', 'TTT')


def load_grid(source: Path) -> dict:
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(code, workload) for code in CODES for workload in WORKLOAD_ORDER}
    cells = {}
    for row in rows:
        key = (row['topology_code'], row['workload'])
        if key in cells:
            raise ValueError(f'duplicate topology cell: {key}')
        cells[key] = row
    if set(cells) != expected:
        raise ValueError(f'incomplete/unexpected grid: {len(expected-set(cells))} missing, '
                         f'{len(set(cells)-expected)} unexpected')
    means = {}
    qps, capacities = [], []
    for code in CODES:
        qps_row, capacity_row = [], []
        for workload in WORKLOAD_ORDER:
            row = cells[code, workload]
            mean = float(row['mean_selectivity'])
            if not math.isfinite(mean) or not 0 <= mean <= 1:
                raise ValueError(f'invalid selectivity: {code}, {workload}')
            if workload in means and not math.isclose(means[workload], mean, rel_tol=1e-9):
                raise ValueError(f'inconsistent selectivity: {workload}')
            means[workload] = mean
            if row['status'] == 'complete':
                throughput, recall = float(row['warm_median_qps']), float(row['recall_min'])
                capacity = int(row['lsearch'])
                if (not math.isfinite(throughput) or throughput <= 0 or
                        not math.isfinite(recall) or not .90 <= recall <= 1 or capacity <= 0):
                    raise ValueError(f'invalid Recall crossing: {code}, {workload}')
            elif row['status'] == 'unavailable':
                # NC must be supported by measured Recall, not a missing input.
                maximum = float(row['max_recall'])
                if not math.isfinite(maximum) or not 0 <= maximum < .90:
                    raise ValueError(f'NC without sub-target measured Recall: {code}, {workload}')
                if row['warm_median_qps'] or row['lsearch'] or row['recall_min']:
                    raise ValueError(f'NC contains a crossing: {code}, {workload}')
                throughput = capacity = math.nan
            else:
                raise ValueError(f'unfinished cell: {code}, {workload}, {row["status"]}')
            qps_row.append(throughput)
            capacity_row.append(capacity)
        qps.append(qps_row)
        capacities.append(capacity_row)
    return {'qps': qps, 'capacities': capacities,
            'mean_selectivities': [means[w] for w in WORKLOAD_ORDER]}


def plot(source: Path, output: Path) -> dict:
    grid = load_grid(source)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    qps = np.array(grid['qps'])
    capacities = np.array(grid['capacities'])
    if not np.isfinite(qps).any(axis=0).all():
        raise ValueError('every workload needs a crossing to define its measured best')
    relative = qps / np.nanmax(qps, axis=0)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    panels = (
        (relative, 'topology_relative_qps', 'Throughput across the complete topology grid',
         'QPS / best measured QPS in the batch', [0.003, .01, .1, 1]),
        (capacities, 'topology_search_budget', 'Search capacity at the measured Recall crossing',
         'Smallest measured crossing capacity', [400, 1000, 10000, 100000, 600000]),
    )
    artifacts = []
    for data, name, title, color_label, ticks in panels:
        fig, ax = plt.subplots(figsize=(7, 4.15), layout='constrained')
        cmap = plt.get_cmap('YlGnBu').copy()
        cmap.set_bad('#dddddd')
        lower, upper = float(np.nanmin(data)), float(np.nanmax(data))
        if lower == upper:
            lower, upper = lower / 2, upper * 2
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap,
                       norm=LogNorm(vmin=lower, vmax=upper), aspect='auto')
        ax.set_xticks(range(len(WORKLOAD_ORDER)),
                      [f'{100*s:.2f}' for s in grid['mean_selectivities']])
        ax.set_yticks(range(len(CODES)), [display_code(c) for c in CODES])
        ax.set_xlabel('Batch-mean selectivity (%)')
        ax.set_title(title, pad=8)
        ax.set_xticks(np.arange(-.5, len(WORKLOAD_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(CODES), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.7)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.axhline(6.5, color='black', linewidth=1.4)
        for i in range(len(CODES)):
            for j in range(len(WORKLOAD_ORDER)):
                if not np.isfinite(data[i, j]):
                    ax.text(j, i, 'NC', ha='center', va='center', fontsize=6.5, color='#555555')
        for j in range(len(WORKLOAD_ORDER)):
            # Ties, if present in a future frozen table, all receive a border.
            for i in np.flatnonzero(qps[:, j] == np.nanmax(qps[:, j])):
                ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False,
                                       edgecolor='#b22222', linewidth=1.4))
        cb = fig.colorbar(im, ax=ax, pad=.025, shrink=.88)
        cb.set_label(color_label)
        ticks = [value for value in ticks if lower <= value <= upper]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{value:g}' for value in ticks])
        for extension in ('pdf', 'png'):
            path = output / f'{name}.{extension}'
            options = {'metadata': {'CreationDate': None, 'ModDate': None}} if extension == 'pdf' else {'dpi': 180}
            fig.savefig(path, **options)
            artifacts.append({'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)
    manifest = {
        'schema_version': 1,
        'input': str(source.resolve()),
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows': len(CODES)*len(WORKLOAD_ORDER),
        'crossings': int(np.isfinite(qps).sum()),
        'no_crossings': int(np.isnan(qps).sum()),
        'recall_target': .90,
        'row_codes': list(CODES),
        'workloads': list(WORKLOAD_ORDER),
        'mean_selectivities': grid['mean_selectivities'],
        'normalization': 'Each QPS is divided by the largest measured crossing QPS in its batch.',
        'border': 'Best measured QPS selected in hindsight; not a significance indicator.',
        'comparison': 'Bases use matched entry providers; upper replacements hold base/provider fixed.',
        'measurement': 'One cold and two warm repeats; smallest measured all-warm Recall crossing.',
        'color_scale': 'Logarithmic; grey NC cells have no throughput or crossing capacity.',
        'matplotlib_version': matplotlib.__version__,
        'outputs': artifacts,
    }
    (output / 'topology_plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('factorial_csv', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    manifest = plot(args.factorial_csv, args.output_dir)
    print(f"Plotted {manifest['rows']} cells: {manifest['crossings']} crossings, "
          f"{manifest['no_crossings']} NCs; {args.output_dir}")


if __name__ == '__main__':
    main()
