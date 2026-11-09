#!/usr/bin/env python3
"""Rebuild the 72-row summary with an explicit base-vector reporting scope.

UNG/PTNC use the separate sizeof/capacity accounting. Other algorithms retain
their recorded RSS metric, minus one base-vector payload; these rows are marked
as estimates, never presented as native sizeof measurements. Repeated runs are
idempotent: values always come from original measurement records.
"""
import csv
import json
from pathlib import Path
import struct

from summarize_index_build import ALGORITHMS, DATASETS, GIB, write_csv


def load_lng_corrections(output):
    path = output/'ung_lng_memory_details.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        records = {r['dataset']:r for r in csv.DictReader(f)}
    if set(records) != set(DATASETS):
        raise ValueError('LNG accounting must cover all eight UNG datasets')
    for dataset, record in records.items():
        audit = json.loads(Path(record['memory_source']).read_text())
        stat = Path(audit['source_file']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (audit['source_file_bytes'], audit['source_mtime_ns']):
            raise ValueError(f'{dataset}: LNG changed; rerun measure_ung_lng_memory.py')
        if int(record['lng_memory_bytes']) != audit['components']['bytes']:
            raise ValueError(f'{dataset}: inconsistent LNG byte count')
    return records


def update_summary(include_vectors=False, excluded_vector_algorithms=()):
    excluded_vector_algorithms = set(excluded_vector_algorithms)
    unknown = excluded_vector_algorithms.difference(ALGORITHMS)
    if unknown:
        raise ValueError(f'Unknown algorithms: {sorted(unknown)}')
    workspace = Path(__file__).resolve().parents[3]
    output = workspace/'FilterVectorResult/index_build_summary'
    summary = output/'memory_and_build_summary.csv'
    with summary.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    original = [dict(r) for r in rows]
    lng_records = load_lng_corrections(output)
    if lng_records:
        backup = output/'memory_and_build_summary_before_adding_lng.csv'
        if not backup.exists():
            backup.write_bytes(summary.read_bytes())
    if excluded_vector_algorithms:
        backup = output/'memory_and_build_summary_before_mixed_vector_scope.csv'
        if not backup.exists():
            backup.write_bytes(summary.read_bytes())
    if include_vectors:
        backup = output/'memory_and_build_summary_before_including_vectors.csv'
        if not backup.exists():
            backup.write_bytes(summary.read_bytes())
    with (output/'loaded_memory_details.csv').open(encoding='utf-8-sig', newline='') as f:
        rss = {(r['dataset'], r['algorithm']): r for r in csv.DictReader(f)}
    with (output/'search_required_memory_details.csv').open(encoding='utf-8-sig', newline='') as f:
        counted = {(r['dataset'], r['algorithm']): r for r in csv.DictReader(f)}
    assert len(rows) == len(rss) == 72
    assert len(counted) == 16, 'Complete all UNG/PTNC counts before publishing'
    vector_sizes = {}
    for dataset in DATASETS:
        path = workspace/'FilterVectorData'/dataset/f'{dataset}_base.bin'
        with path.open('rb') as f:
            n, dim = struct.unpack('<II', f.read(8))
        vector_sizes[dataset] = dict(num_points=n, dim=dim, dtype='float32', bytes=n*dim*4, source=str(path))
    for field in ('memory_method', 'excluded_vector_bytes', 'included_vector_bytes',
                  'index_structure_bytes', 'memory_scope'):
        if field not in fields:
            fields = [*fields, field]
    details = []
    prefix = ('index_memory_mixed_scope' if excluded_vector_algorithms else
              'index_memory_with_vectors' if include_vectors else 'index_structure_memory')
    detail_dir = output/f'{prefix}_details'
    detail_dir.mkdir(exist_ok=True)
    for row in rows:
        key = row['dataset'], row['algorithm']
        vectors = vector_sizes[row['dataset']]['bytes']
        if key in counted:
            c = counted[key]
            assert vectors == int(c['vector_bytes'])
            size = int(c['index_structure_bytes'])
            method = 'search_sizeof_capacity_excluding_vectors'
            source = c['memory_source']
            total = int(c['search_required_memory_bytes'])
        else:
            raw = rss[key]
            total = int(raw['index_rss_delta_bytes'])
            size = total - vectors
            method = 'rss_delta_minus_base_vectors'
            source = raw['source']
        lng_record = lng_records.get(row['dataset']) if row['algorithm']=='UNG' else None
        lng_bytes = int(lng_record['lng_memory_bytes']) if lng_record else 0
        # Historical pruned probes removed LNG. Retain their raw measurements
        # as provenance and add one independently measured native LNG object.
        size += lng_bytes
        total += lng_bytes
        assert size > 0
        structure_bytes = size
        include_row_vectors = include_vectors and key[1] not in excluded_vector_algorithms
        scope = 'index_structures_and_stored_vectors' if include_row_vectors else 'index_structures_only'
        if include_row_vectors:
            size = total
            method = ('search_sizeof_capacity_including_vectors' if key in counted
                      else 'rss_delta_including_vectors')
        excluded_vectors = 0 if include_row_vectors else vectors
        included_vectors = vectors if include_row_vectors else 0
        record = dict(dataset=key[0], algorithm=key[1], index_structure_bytes=structure_bytes,
                      index_structure_gib=structure_bytes/GIB, excluded_vector_bytes=excluded_vectors,
                      included_vector_bytes=included_vectors, loaded_memory_bytes=size,
                      loaded_memory_gib=size/GIB, memory_scope=scope,
                      total_before_excluding_vectors_bytes=total, memory_method=method,
                      measurement_source=source, lng_memory_bytes=lng_bytes,
                      lng_memory_source=lng_record['memory_source'] if lng_record else '')
        audit = detail_dir/f'{key[0]}_{key[1]}.json'
        audit.write_text(json.dumps(record, indent=2)+'\n')
        # Keep unaffected rows' existing source links when applying an exception.
        summary_source = (row['memory_source'] if excluded_vector_algorithms and
                          key[1] not in excluded_vector_algorithms else str(audit))
        row.update(loaded_memory_bytes=str(size), loaded_memory_gib=str(size/GIB),
                   memory_method=method, excluded_vector_bytes=str(excluded_vectors), memory_source=summary_source,
                   included_vector_bytes=str(included_vectors), index_structure_bytes=str(structure_bytes),
                   memory_scope=scope)
        details.append(record)
    allowed = {'loaded_memory_bytes', 'loaded_memory_gib', 'memory_source', 'memory_method',
               'excluded_vector_bytes', 'included_vector_bytes', 'index_structure_bytes', 'memory_scope'}
    for before, after in zip(original, rows):
        assert all(after[k] == v for k,v in before.items() if k not in allowed)
    write_csv(summary, rows, fields)
    write_csv(output/f'{prefix}_details.csv', details)
    write_csv(output/f'{prefix}_gib.csv', [dict(dataset=d, **{
        a: next(r['loaded_memory_gib'] for r in details if r['dataset']==d and r['algorithm']==a)
        for a in ALGORITHMS}) for d in DATASETS])
    if lng_records:
        by_key={(r['dataset'],r['algorithm']):r for r in details}
        write_csv(output/'search_required_memory_gib.csv',[dict(dataset=d, **{
            a:by_key[d,a]['loaded_memory_gib'] for a in ('UNG','PTNC')}) for d in DATASETS])
        comparison_path=output/'ptnc_vs_ung_memory_analysis.csv'
        if comparison_path.exists():
            with comparison_path.open(encoding='utf-8-sig',newline='') as f:
                comparison=list(csv.DictReader(f))
            for item in comparison:
                d=item['dataset']
                item.setdefault('ung_structure_without_lng_gib',item['ung_structure_gib'])
                item['ung_lng_gib']=int(lng_records[d]['lng_memory_bytes'])/GIB
                item['ung_structure_gib']=by_key[d,'UNG']['index_structure_gib']
                item['ptnc_structure_gib']=by_key[d,'PTNC']['index_structure_gib']
                item['ptnc_minus_ung_gib']=item['ptnc_structure_gib']-item['ung_structure_gib']
            write_csv(comparison_path,comparison)
        scope_path=output/'search_required_file_scope.csv'
        if scope_path.exists():
            with scope_path.open(encoding='utf-8-sig',newline='') as f:
                scope_rows=list(csv.DictReader(f))
            for item in scope_rows:
                if item['algorithm']=='UNG' and Path(item['path']).name=='lng_out_neighbors.dat':
                    item['status']='included_search_state'
                    item['role']='LNG outgoing adjacency; incoming adjacency and node arrays reconstructed at load, shared topology counted once'
            write_csv(scope_path,scope_rows)
        raw_method_path=output/'search_required_memory_methodology.json'
        if raw_method_path.exists():
            raw_method=json.loads(raw_method_path.read_text())
            raw_method['ung_lng_addendum']=str(output/'ung_lng_memory_details.csv')
            raw_method['raw_probe_scope']='Historical pruned-core measurement; complete UNG summary adds the separately measured LNG graph.'
            raw_method_path.write_text(json.dumps(raw_method,indent=2)+'\n')
    (output/f'{prefix}_methodology.json').write_text(json.dumps(dict(
        scope=('Algorithm-specific vector scope; see includes_base_vectors_by_algorithm.' if excluded_vector_algorithms else
               'Index structures, filter support and one stored float32 base-vector array per index.'
               if include_vectors else 'Index structures and filter support only; excludes one original float32 base-vector array for every algorithm.'),
        includes_base_vectors_by_algorithm={a: include_vectors and a not in excluded_vector_algorithms for a in ALGORITHMS},
        excluded_vector_algorithms=sorted(excluded_vector_algorithms),
        units='GiB = 2^30 bytes', rows=72, vector_sizes=vector_sizes,
        counted_algorithms=['UNG','PTNC'],
        ung_lng_accounting='Complete UNG includes one LNG graph: in/out adjacency, node arrays, degree arrays and shared control. Historical descendant/coverage caches remain excluded.',
        lng_measurements=str(output/'ung_lng_memory_details.csv') if lng_records else '',
        counted_method='sizeof objects + loaded container capacities + native CRoaring allocations, with unnecessary alternate/build/statistics structures excluded; validated by identical search results after pruning.',
        remaining_algorithms=('Recorded RSS delta including stored base vectors.' if include_vectors
                              else 'Recorded RSS delta minus N*D*sizeof(float).')
                              + ' This is an estimate, includes remaining loader/allocator overhead, and is not a sizeof/capacity count.',
        derived_routing_data='Centroids used for index routing remain part of index structures.',
        duplicate_exports='Redundant reordered_vecs.fvecs exports are not loaded/counted; PTNC shares one vector store with its base index.',
        build_times_preserved=True,
    ), indent=2)+'\n')
    print(f'Updated 72 rows; vectors {"included" if include_vectors else "excluded"}, exceptions={sorted(excluded_vector_algorithms)}; preserved all build times.')


def main():
    update_summary(include_vectors=False)


if __name__ == '__main__':
    main()
