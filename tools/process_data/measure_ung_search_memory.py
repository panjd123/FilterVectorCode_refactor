#!/usr/bin/env python3
"""Count search-required UNG/PTNC storage using sizeof and loaded capacities.

Uses the existing compiled search implementation, without modifying indexes.
For each index, checks 12 deterministic queries before/after removing optional
state. Reports retain the previous RSS measurements separately. Existing build
times and all non-selected summary rows are preserved byte-for-byte as cells.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from summarize_index_build import ALGORITHMS, DATASETS, GIB, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--algorithms', nargs='+', choices=['UNG', 'PTNC'], default=['UNG', 'PTNC'])
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--reuse-probe', action='store_true')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    workspace = Path(__file__).resolve().parents[3]
    project = workspace / 'FilterVectorCode_multilevel_special'
    output = workspace / 'FilterVectorResult/index_build_summary'
    probes = output / 'search_memory_probes'
    probes.mkdir(exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = {k: v for k, v in os.environ.items() if not k.startswith('UNG_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = ':'.join([str(workspace/'openblas/lib'), str(workspace/'boost_1_89_0/lib'),
                                     str(workspace/'zlib/lib'), env.get('LD_LIBRARY_PATH', '')])
    binary = probes / 'UNG_search_memory_probe'
    if not args.reuse_probe or not binary.exists():
        target = project/'build_ung_multilevel_trie_gpu/apps/CMakeFiles/search_UNG_index.dir'
        flags = dict(re.findall(r'^(CXX_\w+) = (.*)$', (target/'flags.make').read_text(), re.M))
        obj = probes/'UNG_search_memory_probe.o'
        compile_cmd = ['/usr/bin/c++']
        for key in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'):
            compile_cmd += shlex.split(flags.get(key, ''))
        compile_cmd += ['-Wno-error', '-c', str(Path(__file__).with_name('ung_search_memory_probe.cpp')), '-o', str(obj)]
        link_cmd = shlex.split((target/'link.txt').read_text())
        link_cmd = [str(obj) if item.endswith('.cpp.o') else item for item in link_cmd]
        link_cmd[link_cmd.index('-o')+1] = str(binary)
        with (probes/'compile.log').open('w') as log:
            for command in (compile_cmd, link_cmd):
                subprocess.run(command, cwd=target.parent.parent, env=env, stdout=log, stderr=log, check=True)
        (probes/'build_commands.json').write_text(json.dumps(dict(compile=compile_cmd, link=link_cmd), indent=2)+'\n')
    destination = output/'search_required_memory_details.csv'
    existing = {}
    if destination.exists():
        with destination.open(encoding='utf-8-sig', newline='') as f:
            existing = {(r['dataset'], r['algorithm']): r for r in csv.DictReader(f)}

    def measure(dataset, algorithm):
        log_path = probes/f'{dataset}_{algorithm}.log'
        folder = workspace/'FilterVectorResult'/dataset/'index'/ALGORITHMS[algorithm]/'index_files'
        command = [str(binary), str(folder), dataset, algorithm]
        print(f'Loading/counting {dataset} {algorithm}', flush=True)
        with log_path.open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
        log_text = log_path.read_text()
        m = re.search(r'SEARCH_MEMORY bytes=(\d+) vector_bytes=(\d+) queries_verified=(\d+) result_pairs_verified=(\d+) neighbor_list_sizeof=(\d+)', log_text)
        if not m:
            raise RuntimeError(f'Missing result: {log_path}')
        size, vectors, queries, pairs, neighbor_size = map(int, m.groups())
        components = [dict(status=status, component=name, bytes=int(value))
                      for status, name, value in re.findall(r'MEMORY_COMPONENT (included|excluded) (\S+) (\d+)', log_text)]
        assert sum(c['bytes'] for c in components if c['status']=='included') == size
        component_file = probes/f'{dataset}_{algorithm}.json'
        component_file.write_text(json.dumps(dict(dataset=dataset, algorithm=algorithm, total_bytes=size,
            vector_bytes=vectors, index_structure_bytes=size-vectors, queries_verified=queries,
            result_pairs_verified=pairs, neighbor_list_sizeof=neighbor_size, components=components,
            command=command), indent=2)+'\n')
        return dict(dataset=dataset, algorithm=algorithm, search_required_memory_bytes=size,
                    search_required_memory_gib=size/GIB, vector_bytes=vectors,
                    index_structure_bytes=size-vectors, queries_verified=queries,
                    result_pairs_verified=pairs, memory_source=str(component_file))

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(measure, d, a): (d, a) for d in args.datasets for a in args.algorithms
                   if args.refresh or (d, a) not in existing}
        for future in as_completed(futures):
            key = futures[future]
            existing[key] = future.result()
            ordered = [existing[d,a] for d in DATASETS for a in ('UNG','PTNC') if (d,a) in existing]
            write_csv(destination, ordered)
            print(f'Counted {key[0]} {key[1]}: {float(existing[key]["search_required_memory_gib"]):.6f} GiB; query results identical', flush=True)

    summary = output/'memory_and_build_summary.csv'
    original_bytes = summary.read_bytes()
    with summary.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    backup = output/'memory_and_build_summary_before_search_sizeof.csv'
    if not backup.exists():
        backup.write_bytes(original_bytes)
    before_cells = [dict(r) for r in rows]
    if 'memory_method' not in fields:
        fields = [*fields, 'memory_method']
    selected = {(d,a) for d in args.datasets for a in args.algorithms}
    for row in rows:
        key = row['dataset'], row['algorithm']
        if key in selected:
            record = existing[key]
            structure_bytes = int(record['index_structure_bytes'])
            include_vectors = row.get('memory_scope') == 'index_structures_and_stored_vectors'
            size = int(record['search_required_memory_bytes']) if include_vectors else structure_bytes
            row['loaded_memory_bytes'] = str(size)
            row['loaded_memory_gib'] = str(size/GIB)
            row['memory_source'] = str(record['memory_source'])
            row['memory_method'] = ('search_sizeof_capacity_including_vectors' if include_vectors
                                    else 'search_sizeof_capacity_excluding_vectors')
            if 'excluded_vector_bytes' in fields:
                row['excluded_vector_bytes'] = '0' if include_vectors else str(record['vector_bytes'])
            if 'included_vector_bytes' in fields:
                row['included_vector_bytes'] = str(record['vector_bytes']) if include_vectors else '0'
            if 'index_structure_bytes' in fields:
                row['index_structure_bytes'] = str(structure_bytes)
        else:
            row.setdefault('memory_method', 'rss_delta')
    allowed = {'loaded_memory_bytes', 'loaded_memory_gib', 'memory_source', 'excluded_vector_bytes',
               'included_vector_bytes', 'index_structure_bytes'}
    for before, after in zip(before_cells, rows):
        for field in fields:
            if field == 'memory_method':
                continue
            assert before[field] == after[field] or ((before['dataset'],before['algorithm']) in selected and field in allowed)
    write_csv(summary, rows, fields)
    includes_vectors = all(r.get('memory_scope') == 'index_structures_and_stored_vectors'
                          for r in rows if (r['dataset'],r['algorithm']) in selected)
    wide_metric = 'search_required_memory_bytes' if includes_vectors else 'index_structure_bytes'
    write_csv(output/'search_required_memory_gib.csv', [dict(dataset=d, **{
        a: int(existing[d,a][wide_metric])/GIB if (d,a) in existing else ''
        for a in ('UNG','PTNC')}) for d in DATASETS])
    (output/'search_required_memory_methodology.json').write_text(json.dumps(dict(
        algorithms=['UNG','PTNC'], method='sizeof objects + actual container capacities + native CRoaring container allocations',
        includes_vectors=includes_vectors, vector_copies_in_main_summary=int(includes_vectors), vector_bytes_in_details_only=not includes_vectors, includes_query_scratch=False,
        includes_allocator_metadata=False, includes_rss=False, build_times_preserved=True,
        other_algorithms='Unchanged RSS measurements; see loaded_memory_methodology.json. Do not interpret as sizeof counts.',
        route='force_use_alg=1, containment, NeighborList; UNG optimized_lng; PTNC trie + free_state blocks, optional router/heavy/HNSW disabled',
        route_statistics='Disabled with UNG_DISABLE_ENTRY_ROUTE_STATS=1; diagnostic bitmaps are excluded.',
        validation='12 real-index vector/label queries per index (full and last-label predicates), K=10 Lsearch=64, identical IDs and distances before/after pruning.',
        backup=str(backup), component_reports=str(probes),
        probe_source_sha256=hashlib.sha256(Path(__file__).with_name('ung_search_memory_probe.cpp').read_bytes()).hexdigest(),
        probe_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        production_loader='Unchanged. It still reads and retains additional compatibility/diagnostic data. Exclusion from this report is not permission to delete a file required by the existing loader.',
    ), indent=2)+'\n')
    if (output/'ung_lng_memory_details.csv').exists():
        # Raw historical probes exclude LNG; publish the complete baseline
        # through the shared, idempotent addendum rather than dropping it again.
        from exclude_vectors_from_memory_summary import update_summary
        include_any = any(r.get('memory_scope')=='index_structures_and_stored_vectors' for r in rows)
        excluded = {r['algorithm'] for r in rows if r.get('memory_scope')=='index_structures_only'}
        update_summary(include_vectors=include_any, excluded_vector_algorithms=excluded)
        with summary.open(encoding='utf-8-sig', newline='') as f:
            published = {(r['dataset'],r['algorithm']):r for r in csv.DictReader(f)}
        write_csv(output/'search_required_memory_gib.csv', [dict(dataset=d, **{
            a: published[d,a]['loaded_memory_gib'] for a in ('UNG','PTNC')}) for d in DATASETS])
        methodology_path=output/'search_required_memory_methodology.json'
        methodology=json.loads(methodology_path.read_text())
        methodology['ung_lng_addendum']=str(output/'ung_lng_memory_details.csv')
        methodology['raw_probe_scope']='Historical pruned-core measurement; complete UNG summary adds the separately measured LNG graph.'
        methodology_path.write_text(json.dumps(methodology,indent=2)+'\n')
    print(f'Updated memory columns only in {summary}', flush=True)


if __name__ == '__main__':
    main()
