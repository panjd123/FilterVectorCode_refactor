#!/usr/bin/env python3
"""Reproduce query composition and Recall strata from frozen crossing records.

This is a source-only reanalysis, not another search experiment. Each method
keeps its own batch-selected capacity. Histograms are over the same queries
in each of two warm executions; no subgroup QPS or independent repetitions
are inferred. Selectivity comes from exact label coverage, never CandSize.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from summarize_base_topology_factorial import WORKLOAD_ORDER
from summarize_warm_repeat_ranges import read_csv, validate_points

N = 602453
BATCH_SIZE = 1000
SIZE_STRATA = ('0', '1', '>=2')
SELECTIVITY_UPPER = (.005, .01, .05, .10, .30, .60, .80, .95, 1.)
SELECTIVITY_LABELS = ('(0,0.5]', '(0.5,1]', '(1,5]', '(5,10]', '(10,30]',
                      '(30,60]', '(60,80]', '(80,95]', '(95,100]')
FIXED = ('L', 'T', 'TL', 'TLT')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def size_stratum(size: int) -> str:
    if size < 0:
        raise ValueError('negative query-label count')
    return str(size) if size < 2 else '>=2'


def selectivity_bin(value: float) -> int:
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('query selectivity must be in (0,1]')
    return bisect.bisect_left(SELECTIVITY_UPPER, value)


def recall_hits(value: str) -> int:
    recall = float(value)
    if not math.isfinite(recall) or not 0 <= recall <= 1:
        raise ValueError('Recall must be finite and in [0,1]')
    hits = round(10 * recall)
    if not math.isclose(10 * recall, hits, rel_tol=0, abs_tol=1e-7):
        raise ValueError('off-grid Recall@10')
    return hits


def validate_coverage(rows: list[dict], *, n: int = N,
                      workloads=WORKLOAD_ORDER, batch_size: int = BATCH_SIZE) -> dict:
    coverage, predicates = {}, {}
    for row in rows:
        key = row['workload'], int(row['QueryID'])
        if key in coverage:
            raise ValueError(f'duplicate coverage key: {key}')
        labels = tuple(int(x) for x in row['labels'].split(',') if x)
        size, eligible = int(row['QuerySize']), int(row['eligible_count'])
        if len(labels) != size or labels != tuple(sorted(set(labels))):
            raise ValueError(f'noncanonical or duplicate labels: {key}')
        # All queries in this frozen corpus have >=20 eligible vectors. The
        # >=K test is the needed contract for the Recall denominator used here.
        if not 10 <= eligible <= n or (size == 0 and eligible != n):
            raise ValueError(f'invalid eligible count or empty predicate: {key}')
        if labels in predicates and predicates[labels] != eligible:
            raise ValueError(f'inconsistent coverage of the same predicate: {key}')
        predicates[labels] = eligible
        selectivity = float(row['true_selectivity'])
        if not math.isclose(selectivity, eligible / n, rel_tol=0, abs_tol=1e-14):
            raise ValueError(f'coverage/selectivity mismatch: {key}')
        coverage[key] = dict(size=size, eligible=eligible, selectivity=selectivity,
                             size_stratum=size_stratum(size),
                             selectivity_bin=SELECTIVITY_LABELS[selectivity_bin(selectivity)])
    expected = {(w, q) for w in workloads for q in range(batch_size)}
    if set(coverage) != expected:
        raise ValueError('missing, extra, or out-of-range coverage query IDs')
    return coverage


def aggregate_records(rows, coverage: dict, points: list[dict],
                      *, batch_size: int = BATCH_SIZE) -> tuple[dict, list[dict]]:
    """Keep integer hit histograms so regrouping has no averaging ambiguity."""
    expected = {(p['topology_code'], p['workload'], repeat): p
                for p in points for repeat in (1, 2)}
    if len(expected) != 2 * len(points):
        raise ValueError('duplicate operating points')
    seen = defaultdict(set)
    histograms = defaultdict(Counter)
    for row in rows:
        code, workload = row['topology_code'], row['workload']
        repeat, qid = int(row['repeat']), int(row['QueryID'])
        key = code, workload, repeat
        point = expected.get(key)
        if point is None:
            raise ValueError(f'query row has no selected warm crossing: {key}')
        if row['method'] != point['method'] or int(row['lsearch']) != int(point['lsearch']):
            raise ValueError(f'wrong method or capacity: {key}')
        if qid in seen[key]:
            raise ValueError(f'duplicate query ID: {key}, {qid}')
        metadata = coverage.get((workload, qid))
        if metadata is None or int(row['QuerySize']) != metadata['size']:
            raise ValueError(f'query metadata join failed: {key}, {qid}')
        seen[key].add(qid)
        hits = recall_hits(row['Recall'])
        for kind, stratum in [('size', metadata['size_stratum']),
                              ('selectivity', metadata['selectivity_bin']), ('all', 'all')]:
            histograms[code, workload, repeat, kind, stratum][hits] += 1
    checks = []
    for key, point in expected.items():
        if seen[key] != set(range(batch_size)):
            raise ValueError(f'incomplete selected batch: {key}')
        hist = histograms[(*key, 'all', 'all')]
        recall = sum(hits * count for hits, count in hist.items()) / (10 * batch_size)
        target = float(point[f'warm_{key[2]}_recall'])
        if not math.isclose(recall, target, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'query Recall does not reconstruct batch crossing: {key}')
        checks.append(dict(topology_code=key[0], workload=key[1], repeat=key[2],
                           count=batch_size, reconstructed_recall=recall,
                           recorded_recall=target, absolute_error=abs(recall-target)))
    return dict(histograms), checks


def check_reference(histograms: dict, rows: list[dict], points: list[dict]) -> None:
    """Cross-check the earlier, independently extracted size-stratum table."""
    expected = {(p['topology_code'], p['workload'], r, s)
                for p in points for r in (1, 2) for s in SIZE_STRATA}
    seen = set()
    for row in rows:
        key = row['topology_code'], row['workload'], int(row['repeat']), row['query_size_stratum']
        if key in seen or key not in expected:
            raise ValueError(f'duplicate or unexpected reference stratum: {key}')
        seen.add(key)
        hist = histograms.get((*key[:3], 'size', key[3]), Counter())
        if any(hist.get(i, 0) != int(row[f'recall_hist_{i/10:.1f}']) for i in range(11)):
            raise ValueError(f'query histogram disagrees with prior extraction: {key}')
        if int(row['count']) != sum(hist.values()):
            raise ValueError(f'reference count mismatch: {key}')
    if seen != expected:
        raise ValueError('missing reference strata, including empty strata')


def summarize_hist(hist: Counter) -> dict:
    count = sum(hist.values())
    return dict(count=count, mean_recall=(sum(k*v for k, v in hist.items())/(10*count)
                                         if count else ''),
                recall_zero=hist.get(0, 0), recall_below_0p9=sum(hist.get(i, 0) for i in range(9)),
                **{f'recall_hist_{i/10:.1f}': hist.get(i, 0) for i in range(11)})


def summarize_strata(histograms: dict, coverage: dict, cells: list[dict], kind: str) -> list[dict]:
    result = []
    strata = SIZE_STRATA if kind == 'size' else SELECTIVITY_LABELS
    field = 'size_stratum' if kind == 'size' else 'selectivity_bin'
    counts = Counter((workload, row[field]) for (workload, _), row in coverage.items())
    for cell in cells:
        code, workload = cell['topology_code'], cell['workload']
        for repeat in (1, 2):
            for stratum in strata:
                n = counts[workload, stratum]
                base = dict(topology_code=code, method=cell['method'], workload=workload,
                            lsearch=cell['lsearch'], repeat=repeat, stratum=stratum,
                            workload_stratum_count=n)
                if cell['status'] != 'complete':
                    metrics = {key: '' for key in summarize_hist(Counter())}
                    result.append(dict(base, status='no_crossing', **metrics))
                    continue
                hist = histograms.get((code, workload, repeat, kind, stratum), Counter())
                if sum(hist.values()) != n:
                    raise ValueError(f'stratum/query coverage count mismatch: {code}, {workload}')
                result.append(dict(base, status='complete' if n else 'empty_stratum',
                                   **summarize_hist(hist)))
    return result


def range_text(values, decimals=2) -> str:
    low, high = min(values), max(values)
    fmt = (lambda x: str(int(x))) if decimals == 0 else (lambda x: f'{x:.{decimals}f}')
    return fmt(low) if fmt(low) == fmt(high) else fmt(low)+'--'+fmt(high)


def make_table(rows: list[dict], cells: list[dict]) -> str:
    lookup = {(r['topology_code'], r['workload'], r['stratum'], r['repeat']): r for r in rows}
    means = {r['workload']: float(r['mean_selectivity']) for r in cells}
    lines = [r'\newcommand{\querySizeRecallTable}{%', r'\begin{table}[H]',
             r'\centering\small',
             r"\caption{Recall within predicate-size strata at each method's "
             r'own batch-selected crossing. Each cell gives mean Recall@10 in percent, '
             r'followed by the number of queries with zero Recall in parentheses. '
             r'Ranges span the two original warm executions. $n$ is the number of '
             r'queries per execution, not a count of independent repetitions. '
             r'TL and TLT denote \texttt{1L[T|L]} and \texttt{2L[T|LT]}; '
             r'L and T are their matched-provider 0L systems. NC has no selected crossing.}',
             r'\label{tab:query-size-recall}', r'\begin{tabular}{rrrllll}', r'\toprule',
             r'Mean sel. & $|Q|$ & $n$ & L & T & TL & TLT \\', r'\midrule']
    for workload in WORKLOAD_ORDER:
        for stratum in SIZE_STRATA:
            n = lookup['L', workload, stratum, 1]['workload_stratum_count']
            if n == 0:
                continue
            vals = []
            for code in FIXED:
                pair = [lookup[code, workload, stratum, repeat] for repeat in (1, 2)]
                if pair[0]['status'] == 'no_crossing':
                    vals.append('NC')
                else:
                    mean = range_text([100 * r['mean_recall'] for r in pair])
                    zeros = range_text([r['recall_zero'] for r in pair], 0)
                    vals.append(f'{mean} ({zeros})')
            size = r'$\geq2$' if stratum == '>=2' else stratum
            lines.append(f'{100*means[workload]:.3f}\\% & {size} & {n} & '
                         +' & '.join(vals)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '}']
    return '\n'.join(lines)+'\n'


def make_selectivity_table(rows: list[dict], cells: list[dict]) -> str:
    lookup = {(r['topology_code'], r['workload'], r['stratum'], r['repeat']): r for r in rows}
    means = {r['workload']: float(r['mean_selectivity']) for r in cells}
    lines = [r'\newcommand{\querySelectivityRecallTable}{%', r'\begin{table}[H]',
             r'\centering\small',
             r'\caption{Recall by exact individual-query selectivity in the two mixed '
             r'batches discussed in the text. Bins are open on the left and closed '
             r'on the right. Cells use the Recall-percent (zero-count) convention '
             r'of Table~\ref{tab:query-size-recall}, at the same batch-selected crossings. '
             r'The source CSV retains all nine workloads, every bin, and all 14 configurations.}',
             r'\label{tab:query-selectivity-recall}', r'\begin{tabular}{rlrllll}', r'\toprule',
             r'Mean sel. & Query sel. (\%) & $n$ & L & T & TL & TLT \\', r'\midrule']
    for workload in ('sel_5', 'sel_30'):
        for stratum in SELECTIVITY_LABELS:
            n = lookup['L', workload, stratum, 1]['workload_stratum_count']
            if n == 0:
                continue
            vals = []
            for code in FIXED:
                pair = [lookup[code, workload, stratum, r] for r in (1, 2)]
                vals.append('NC' if pair[0]['status']=='no_crossing' else
                            range_text([100*r['mean_recall'] for r in pair])+' ('+
                            range_text([r['recall_zero'] for r in pair], 0)+')')
            lines.append(f'{100*means[workload]:.3f}\\% & {stratum} & {n} & '
                         +' & '.join(vals)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '}']
    return '\n'.join(lines)+'\n'


def plot_composition(coverage: dict, cells: list[dict], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    means = {r['workload']: float(r['mean_selectivity']) for r in cells}
    size_counts = Counter((w, r['size_stratum']) for (w, _), r in coverage.items())
    sel_counts = Counter((w, r['selectivity_bin']) for (w, _), r in coverage.items())
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.05, 3.25),
                                  gridspec_kw={'width_ratios': [1, 1.6]})
    y = np.arange(len(WORKLOAD_ORDER))
    left = np.zeros(len(y))
    for stratum, color, label in zip(SIZE_STRATA, ['#555555', '#d48324', '#287e8e'],
                                     [r'$|Q|=0$', r'$|Q|=1$', r'$|Q|\geq2$']):
        values = np.array([size_counts[w, stratum] for w in WORKLOAD_ORDER])
        ax.barh(y, values, left=left, color=color, height=.76, label=label)
        for i, value in enumerate(values):
            if value >= 150:
                ax.text(left[i]+value/2, i, str(value), ha='center', va='center', color='white', fontsize=7)
        left += values
    ax.set_yticks(y, [f'{100*means[w]:.3f}%' for w in WORKLOAD_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 500, 1000])
    ax.set_xlabel('Queries per batch')
    ax.set_ylabel('Batch-mean selectivity')
    ax.set_title('(a) Predicate composition', fontsize=9)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.07), ncol=3,
              frameon=False, handlelength=.9, columnspacing=.8)
    matrix = np.array([[sel_counts[w, s] for s in SELECTIVITY_LABELS] for w in WORKLOAD_ORDER])
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#f2f2f2')
    bx.imshow(np.ma.masked_equal(matrix, 0), cmap=cmap, norm=LogNorm(vmin=1, vmax=1000), aspect='auto')
    for i, j in np.ndindex(matrix.shape):
        v = int(matrix[i, j])
        if v:
            bx.text(j, i, str(v), ha='center', va='center', fontsize=6.4,
                    color='white' if v >= 90 else 'black')
    bx.set_yticks(y, [f'{100*means[w]:.3f}%' for w in WORKLOAD_ORDER])
    bx.set_xticks(range(len(SELECTIVITY_LABELS)), SELECTIVITY_LABELS, rotation=55, ha='right', fontsize=6.5)
    bx.set_xlabel('Individual-query selectivity bin (%)')
    bx.set_title('(b) Queries by individual selectivity', fontsize=9, pad=22)
    fig.subplots_adjust(left=.13, right=.995, bottom=.27, top=.80, wspace=.39)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output/'query_composition.pdf', output/'query_composition.png']
    fig.savefig(paths[0], metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(paths[1], dpi=220)
    plt.close(fig)
    return paths


def summarize(data: Path, factorial: Path, warm_points: Path, output: Path, figures: Path) -> dict:
    cells, points = read_csv(factorial), read_csv(warm_points)
    validate_points(points, cells)
    inputs = [data/'query_coverage_exact.csv', data/'query_recall.csv.gz',
              data/'query_size_reference.csv']
    frozen = json.loads((data/'input_hashes.json').read_text())
    for p in inputs:
        if sha256(p) != frozen[p.name]:
            raise ValueError(f'frozen input hash mismatch: {p.name}')
    coverage = validate_coverage(read_csv(inputs[0]))
    for workload in WORKLOAD_ORDER:
        exact_mean = sum(r['selectivity'] for (w, _), r in coverage.items() if w == workload)/BATCH_SIZE
        recorded = float(next(r for r in cells if r['workload'] == workload)['mean_selectivity'])
        if not math.isclose(exact_mean, recorded, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f'exact coverage does not reproduce workload mean: {workload}')
    with gzip.open(inputs[1], 'rt', newline='') as stream:
        histograms, checks = aggregate_records(csv.DictReader(stream), coverage, points)
    check_reference(histograms, read_csv(inputs[2]), points)
    size_rows = summarize_strata(histograms, coverage, cells, 'size')
    sel_rows = summarize_strata(histograms, coverage, cells, 'selectivity')
    output.mkdir(parents=True, exist_ok=True)
    outputs = [output/'query_size_summary.csv', output/'query_selectivity_summary.csv',
               output/'repeat_reconstruction.csv', output/'generated_query_strata.tex']
    for p, rows in zip(outputs[:3], [size_rows, sel_rows, checks]):
        write_csv(p, rows)
    outputs[3].write_text(make_table(size_rows, cells)+make_selectivity_table(sel_rows, cells))
    outputs += plot_composition(coverage, cells, figures)
    manifest = dict(scope='Regrouped frozen exported Recall; no new ANN or build benchmark.',
                    coverage_rows=len(coverage), crossing_batches=len(checks),
                    query_records=len(checks)*BATCH_SIZE, prior_size_histograms_checked=len(points)*6,
                    base_size=N, max_reconstruction_error=max(r['absolute_error'] for r in checks),
                    size_summary_rows=len(size_rows), selectivity_summary_rows=len(sel_rows),
                    selection='Each method retains its own globally selected batch-mean Recall crossing.',
                    selectivity='Exact eligible_count / 602453; right-closed bins in (0,1].',
                    repeats='Two executions of the same queries, not independent query samples.',
                    limitations=['No subgroup-matched Recall or subgroup throughput.',
                                 'Exported Recall regrouping does not recompute nearest-neighbor ground truth.',
                                 'Current label coverage does not supply missing historical query-content hashes.'],
                    inputs=[dict(name=p.name, sha256=sha256(p)) for p in inputs+[factorial, warm_points]],
                    script_sha256=sha256(Path(__file__)),
                    outputs=[dict(name=p.name, sha256=sha256(p)) for p in outputs])
    (output/'summary_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--factorial', type=Path, required=True)
    parser.add_argument('--warm-points', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = summarize(args.data_dir, args.factorial, args.warm_points, args.output_dir, args.figure_dir)
    print(f'Validated {manifest["query_records"]} records and {manifest["coverage_rows"]} exact coverages.')


if __name__ == '__main__':
    main()
