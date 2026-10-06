#!/usr/bin/env python3
"""Read-only audit of the two recorded warm repeats, not a new experiment."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path('/home/sunyahui/worktrees/FilterVectorCode_multilevel_special')
PAPER = ROOT / 'docs/papers/multilevel_ung'
OUT = ROOT / 'runs/paper_c1_20261007/warm_repeat_audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = json.loads((PAPER / 'generated_data/manifest.json').read_text())
    inputs = []
    sources = []
    for source in manifest['inputs']:
        p = Path(source['path'])
        if p.name == 'all_points.csv':
            assert sha(p) == source['sha256'], p
            sources.extend(read_csv(p))
            inputs.append(source)
    factorial = PAPER / 'generated_data/factorial_equal_recall.csv'
    inputs.append({'path': str(factorial), 'sha256': sha(factorial)})
    selected = read_csv(factorial)
    rows = []
    raw_inputs = {}
    for cell in selected:
        if cell['status'] != 'complete':
            continue
        candidates = [r for r in sources if
                      r['method'] == cell['method'] and
                      r['workload'] == cell['workload'] and
                      int(r['lsearch']) == int(cell['lsearch']) and
                      math.isclose(float(r['qps_warm_median']), float(cell['warm_median_qps']), rel_tol=1e-10)]
        assert candidates, cell
        paths = {r['summary_path'] for r in candidates}
        assert len(paths) == 1, paths
        source = candidates[0]
        detail_path = Path(source['summary_path']).with_name('search_time_details.csv')
        repeats = [r for r in read_csv(detail_path) if int(r['Lsearch']) == int(cell['lsearch'])]
        assert sorted(int(r['Repeat']) for r in repeats) == [0, 1, 2], detail_path
        warm = sorted([r for r in repeats if int(r['Repeat']) >= 1], key=lambda r: int(r['Repeat']))
        times = [float(r['Time_ms']) for r in warm]
        recalls = [float(r['Avg_Recall']) for r in warm]
        assert all(math.isfinite(t) and t > 0 for t in times)
        assert min(recalls) >= .90
        count = int(source['num_queries'])
        median = statistics.median(times)
        qps = 1000 * count / median
        assert math.isclose(qps, float(cell['warm_median_qps']), rel_tol=1e-10), cell
        assert math.isclose(min(recalls), float(cell['recall_min']), abs_tol=1e-10), cell
        cv = statistics.stdev(times) / statistics.mean(times)
        assert math.isclose(cv, float(source['batch_ms_warm_cv']), abs_tol=1e-10), cell
        raw_inputs[str(detail_path)] = sha(detail_path)
        rows.append({
            'topology_code': cell['topology_code'], 'workload': cell['workload'],
            'mean_selectivity': cell['mean_selectivity'], 'method': cell['method'],
            'lsearch': int(cell['lsearch']), 'warm_1_ms': times[0], 'warm_2_ms': times[1],
            'warm_1_recall': recalls[0], 'warm_2_recall': recalls[1],
            'qps_at_median_time': qps, 'warm_cv': cv,
            'min_observed_qps': 1000 * count / max(times),
            'max_observed_qps': 1000 * count / min(times),
            'detail_path': str(detail_path), 'detail_sha256': sha(detail_path),
        })
    by_key = {(r['topology_code'], r['workload']): r for r in rows}
    workloads = list(dict.fromkeys(c['workload'] for c in selected))
    comparisons = []
    for left_code, right_code in [('T', 'L'), ('TLT', 'L'), ('TLT', 'TL'), ('TLT', 'TLL')]:
        for workload in workloads:
            left, right = by_key.get((left_code, workload)), by_key.get((right_code, workload))
            if left is None or right is None:
                continue
            lower = left['min_observed_qps'] / right['max_observed_qps']
            upper = left['max_observed_qps'] / right['min_observed_qps']
            comparisons.append({
                'comparison': left_code + '/' + right_code, 'workload': workload,
                'median_time_qps_ratio': left['qps_at_median_time'] / right['qps_at_median_time'],
                'min_cross_repeat_ratio': lower, 'max_cross_repeat_ratio': upper,
                'observed_ranges_overlap': lower <= 1 <= upper,
            })
    OUT.mkdir(parents=True, exist_ok=True)
    for name, records in [('warm_repeat_points.csv', rows), ('warm_repeat_comparisons.csv', comparisons)]:
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    report = {
        'scope': 'Descriptive ranges from the original two warm repeats. No new timing run.',
        'not_a_confidence_interval': True,
        'range_definition': 'All four observed cross-repeat QPS ratios; repeats are not treated as paired.',
        'limitations': 'Overlap or separation of these two observed ranges does not establish statistical significance or cross-run reproducibility.',
        'crossings_checked': len(rows), 'nc_cells_excluded': len(selected) - len(rows),
        'all_qps_recall_cv_reconstructed': True,
        'input_summaries': inputs,
        'raw_time_inputs': [{'path': p, 'sha256': h} for p, h in sorted(raw_inputs.items())],
        'script_sha256': sha(Path(__file__)),
        'outputs': [{'path': str(OUT / name), 'sha256': sha(OUT / name)} for name in
                    ['warm_repeat_points.csv', 'warm_repeat_comparisons.csv']],
    }
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Reconstructed', len(rows), 'crossings from original warm timing rows.')
    print('Warm CV median/max:', statistics.median(r['warm_cv'] for r in rows), max(r['warm_cv'] for r in rows))
    for row in comparisons:
        if row['comparison'] in ('TLT/TL', 'TLT/TLL'):
            print(row)


if __name__ == '__main__':
    main()
