#!/usr/bin/env python3
"""Export existing frozen-crossing Recall scalars; never run ANN or build work.

This is a raw-artifact extractor for the published 14x9 Amazon screen, not an
independent exact-neighbor verification. The exported gzip enables later
source-only reanalysis without the original wide query-detail files.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

FIELDS = ['workload', 'method', 'topology_code', 'lsearch', 'repeat',
          'QueryID', 'QuerySize', 'Recall']
CODES = {'L', 'T', 'LL', 'LT', 'TL', 'TT', 'LLL', 'LLT', 'LTL',
         'LTT', 'TLL', 'TLT', 'TTL', 'TTT'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def gzip_records(records):
    target = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=target, mode='wb', compresslevel=9, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
    return target.getvalue()


def extract(repo, output):
    repo, output = repo.resolve(), output.resolve()
    require(not output.is_relative_to(Path('/home/graphdb')), 'output under /home/graphdb is forbidden')
    require(not output.exists(), f'output directory already exists: {output}')
    points_path = repo/'docs/papers/multilevel_ung/generated_data/warm_repeats/warm_repeat_points.csv'
    factorial_path = repo/'docs/papers/multilevel_ung/generated_data/factorial_equal_recall.csv'
    inputs = {}

    def before(path):
        digest = sha(path)
        require(path not in inputs, f'duplicate input path: {path}')
        inputs[path] = {'path': str(path), 'sha256_before': digest, 'bytes': path.stat().st_size}
        return digest

    def after(path, expected):
        actual = sha(path)
        require(actual == expected, f'input changed during read: {path}')
        inputs[path]['sha256_after'] = actual

    points_hash = before(points_path)
    points = read_csv(points_path)
    after(points_path, points_hash)
    factorial_hash = before(factorial_path)
    cells = read_csv(factorial_path)
    after(factorial_path, factorial_hash)
    key = lambda row: (row['topology_code'], row['workload'])
    cell_by_key = {key(row): row for row in cells}
    workloads = {row['workload'] for row in cells}
    require(len(cells) == len(cell_by_key) == 126 and len(workloads) == 9, 'expected unique 14x9 grid')
    require(set(cell_by_key) == {(code, workload) for code in CODES for workload in workloads}, 'incomplete grid')
    require(all(row['status'] in {'complete', 'unavailable'} for row in cells), 'unknown cell status')
    complete = {k for k, row in cell_by_key.items() if row['status'] == 'complete'}
    require(len(points) == len({key(row) for row in points}) == 95, 'expected 95 unique warm points')
    require({key(row) for row in points} == complete, 'warm points do not equal all completed crossings')
    records, checks, strata, label_maps = [], [], [], {}
    for number, point in enumerate(sorted(points, key=lambda row: (row['workload'], row['topology_code'], row['method'])), 1):
        cell = cell_by_key[key(point)]
        require(point['method'] == cell['method'] and int(point['lsearch']) == int(cell['lsearch']), 'crossing method/capacity mismatch')
        times = [float(point[f'warm_{repeat}_ms']) for repeat in (1, 2)]
        recalls = [float(point[f'warm_{repeat}_recall']) for repeat in (1, 2)]
        require(all(math.isfinite(t) and t > 0 for t in times), 'invalid warm times')
        require(all(math.isfinite(r) and .9 <= r <= 1 for r in recalls), 'invalid warm crossing Recall')
        qps = 1000000/statistics.median(times)
        require(close(qps, float(cell['warm_median_qps'])) and close(qps, float(point['qps_at_median_time'])), 'QPS mismatch')
        require(close(min(recalls), float(cell['recall_min'])), 'crossing Recall mismatch')
        original = Path(point['detail_path'])
        try:
            relative = original.relative_to(repo)
        except ValueError:
            require('/runs/' in str(original), f'cannot relocate detail path: {original}')
            relative = Path('runs')/str(original).split('/runs/', 1)[1]
        detail = (repo/relative).with_name('query_details_repeat3.csv').resolve()
        require(detail.is_relative_to(repo/'runs'), f'detail outside repo runs: {detail}')
        digest = before(detail)
        selected = {1: {}, 2: {}}
        with detail.open(newline='') as stream:
            for row in csv.DictReader(stream):
                repeat = int(row['Repeat'])
                if repeat not in selected or int(row['Lsearch']) != int(point['lsearch']):
                    continue
                query, size, recall = int(row['QueryID']), int(row['QuerySize']), float(row['Recall'])
                require(query not in selected[repeat], f'duplicate query ID: {detail}, {repeat}, {query}')
                require(size >= 0 and math.isfinite(recall) and 0 <= recall <= 1, 'invalid query value')
                require(abs(recall-round(recall*10)/10) <= 1e-9, 'off-grid Recall@10')
                selected[repeat][query] = (size, recall)
        after(detail, digest)
        for repeat, queries in selected.items():
            require(set(queries) == set(range(1000)), f'query ID coverage error: {detail}, {repeat}')
            mapping = {query: item[0] for query, item in queries.items()}
            prior = label_maps.setdefault(point['workload'], mapping)
            require(mapping == prior, f'cross-method/repeat QuerySize mismatch: {key(point)}')
            observed = statistics.fmean(value[1] for value in queries.values())
            expected = float(point[f'warm_{repeat}_recall'])
            require(abs(observed-expected) <= 5e-7, f'global Recall mismatch: {detail}, {repeat}')
            checks.append({'workload': point['workload'], 'method': point['method'], 'topology_code': point['topology_code'],
                           'lsearch': int(point['lsearch']), 'repeat': repeat, 'count': 1000,
                           'mean_recall': observed, 'warm_point_recall': expected, 'absolute_difference': abs(observed-expected)})
            for query, (size, recall) in sorted(queries.items()):
                records.append({'workload': point['workload'], 'method': point['method'], 'topology_code': point['topology_code'],
                                'lsearch': int(point['lsearch']), 'repeat': repeat, 'QueryID': query,
                                'QuerySize': size, 'Recall': format(recall, '.17g')})
            for group in ('0', '1', '>=2'):
                values = [recall for size, recall in queries.values() if ('0' if size == 0 else '1' if size == 1 else '>=2') == group]
                row = {name: checks[-1][name] for name in ('workload', 'method', 'topology_code', 'lsearch', 'repeat')}
                row.update({'query_size_stratum': group, 'count': len(values), 'mean_recall': statistics.fmean(values) if values else '',
                            'recall_below_0p9': sum(r < .9 for r in values), 'recall_zero': sum(r == 0 for r in values)})
                row.update({f'recall_hist_{i/10:.1f}': sum(round(r*10) == i for r in values) for i in range(11)})
                strata.append(row)
        if number % 10 == 0:
            print(f'Validated {number}/95 source files', flush=True)
    require(len(records) == 190000 and len(checks) == 190 and len(strata) == 570, 'record count mismatch')
    packed = gzip_records(records)
    require(packed == gzip_records(records), 'gzip output is not deterministic in this environment')
    decoded = list(csv.DictReader(io.StringIO(gzip.decompress(packed).decode('utf-8'))))
    require(len(decoded) == 190000 and list(decoded[0]) == FIELDS, 'export schema/count mismatch')
    for actual, expected in zip(decoded, records):
        require(actual == {k: str(v) for k, v in expected.items()}, 'gzip roundtrip changed a record')
    for path, item in inputs.items():
        item['sha256_final'] = sha(path)
        require(item['sha256_final'] == item['sha256_before'], f'input changed before publication: {path}')
    output.mkdir(parents=True, exist_ok=False)
    (output/'query_recall_records.csv.gz').write_bytes(packed)
    write_csv(output/'repeat_recall_checks.csv', checks)
    write_csv(output/'query_size_strata.csv', strata)
    write_csv(output/'workload_query_sizes.csv', [{'workload': w, 'QueryID': q, 'QuerySize': n} for w, mapping in sorted(label_maps.items()) for q, n in sorted(mapping.items())])
    manifest = {'scope': 'Regroup/export existing Recall scalars; not independent exact-neighbor verification. No ANN/build run.',
                'repo': str(repo), 'selected_crossings': 95, 'nc_cells_excluded': 31, 'warm_batches': 190,
                'query_records': 190000, 'stratum_rows': 570, 'schema': FIELDS,
                'ordering': 'workload, topology_code, method (lexical); repeat 1 then 2; QueryID ascending',
                'gzip': {'mtime': 0, 'filename': '', 'compresslevel': 9, 'csv_line_ending': 'LF',
                         'two_serializations_identical': True, 'full_roundtrip_matches': True,
                         'uncompressed_sha256': hashlib.sha256(gzip.decompress(packed)).hexdigest()},
                'query_ids_per_batch': 'exactly 0..999', 'cross_method_repeat_label_counts_match': True,
                'batch_mean_recall_maximum_error': max(r['absolute_difference'] for r in checks),
                'all_inputs_hash_stable_before_after_and_final': True,
                'inputs': list(inputs.values()), 'script_sha256': sha(Path(__file__)),
                'limitations': ['Global crossing is not a per-stratum guarantee.', 'Two warm repeats are not independent per-query experiments.',
                                'QuerySize is loaded label-vector length, not selectivity; independently audited labels are needed for coverage joins.',
                                'No CandSize or LIGHT_STATS counter interpretation; no timing or subgroup QPS exported.'],
                'outputs': [{'name': name, 'bytes': (output/name).stat().st_size, 'sha256': sha(output/name)} for name in
                            ('query_recall_records.csv.gz', 'repeat_recall_checks.csv', 'query_size_strata.csv', 'workload_query_sizes.csv')]}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Exported 190000 existing query records to {output}; gzip bytes={len(packed)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.repo, args.output)


if __name__ == '__main__':
    main()
