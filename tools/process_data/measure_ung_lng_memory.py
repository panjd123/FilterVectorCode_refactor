#!/usr/bin/env python3
"""Measure the native LNG graph independently of vector storage and old caches."""
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from summarize_index_build import DATASETS, GIB, write_csv


def main():
    workspace = Path(__file__).resolve().parents[3]
    project = workspace/'FilterVectorCode_multilevel_special'
    output = workspace/'FilterVectorResult/index_build_summary'
    probes = output/'lng_memory_probes'
    probes.mkdir(exist_ok=True)
    source = Path(__file__).with_name('ung_lng_memory_probe.cpp')
    target = project/'build_ung_multilevel_trie_gpu/apps/CMakeFiles/search_UNG_index.dir'
    flags = dict(re.findall(r'^(CXX_\w+) = (.*)$', (target/'flags.make').read_text(), re.M))
    binary = probes/'ung_lng_memory_probe'
    command = ['/usr/bin/c++', '-O2', '-std=c++17']
    for key in ('CXX_DEFINES', 'CXX_INCLUDES'):
        command += shlex.split(flags.get(key, ''))
    command += [str(source), '-o', str(binary)]
    with (probes/'compile.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    (probes/'build_commands.json').write_text(json.dumps(dict(compile=command,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), indent=2)+'\n')

    def measure(dataset):
        folder=workspace/'FilterVectorResult'/dataset/'index/UNG/index_files'
        meta=dict(line.split('=',1) for line in (folder/'meta').read_text().splitlines() if '=' in line)
        if meta.get('base_group_topology', 'lng') != 'lng':
            raise ValueError(f'{dataset} UNG is not built with LNG topology')
        adjacency=folder/'lng_out_neighbors.dat'
        stat=adjacency.stat()
        native_command=[str(binary), str(adjacency), meta['num_groups'], meta['LNG_num_edges']]
        result=subprocess.run(native_command, text=True, capture_output=True, check=True)
        (probes/f'{dataset}_UNG.log').write_text(result.stdout+result.stderr)
        values={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)', result.stdout)}
        assert values['bytes']==sum(values[k] for k in ('object_bytes','outer_array_bytes',
            'outgoing_capacity_bytes','incoming_capacity_bytes','degree_capacity_bytes','shared_control_bytes'))
        assert (stat.st_size,stat.st_mtime_ns)==(adjacency.stat().st_size,adjacency.stat().st_mtime_ns)
        audit=probes/f'{dataset}_UNG.json'
        audit.write_text(json.dumps(dict(dataset=dataset, algorithm='UNG',
            method='Native LNG object sizeof plus allocated container capacities; shared object counted once.',
            scope='LNG outgoing/incoming adjacency, outer node arrays, degree arrays and shared ownership control.',
            excludes='Vectors, old descendant/coverage caches, queries and allocator-internal overhead.',
            source_file=str(adjacency),source_file_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
            num_groups=int(meta['num_groups']),components=values,command=native_command),indent=2)+'\n')
        print(f'{dataset}: LNG {values["bytes"]/GIB:.6f} GiB, {values["edges"]} edges verified',flush=True)
        return dict(dataset=dataset,algorithm='UNG',lng_memory_bytes=values['bytes'],
                    lng_memory_gib=values['bytes']/GIB,lng_edges=values['edges'],memory_source=str(audit))

    results={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(measure,d):d for d in DATASETS}
        for future in as_completed(futures):
            results[futures[future]]=future.result()
    write_csv(output/'ung_lng_memory_details.csv',[results[d] for d in DATASETS])


if __name__=='__main__':
    main()
