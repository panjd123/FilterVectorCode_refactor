#!/usr/bin/env python3
"""Include stored vectors for seven baselines; UNG/PTNC report structures only.

Reuses original RSS records and the UNG/PTNC sizeof/capacity measurements.
Preserves construction times and keeps structure/vector components separate.
"""
from pathlib import Path
import struct

from exclude_vectors_from_memory_summary import update_summary
from summarize_index_build import ALGORITHMS, DATASETS, write_csv


def verify_vector_storage():
    workspace = Path(__file__).resolve().parents[3]
    project = workspace/'FilterVectorCode_multilevel_special'
    output = workspace/'FilterVectorResult/index_build_summary'
    specs = {
        'Milvus': ('index.bin', 'Knowhere/project_adapter/common.hpp', 131),
        'ACORN': ('acorn.index', 'ACORN/faiss/impl/index_write.cpp', 778),
        'NaviX': ('vecs.bin', 'UNG/codes/src/navix_index.cpp', 60),
        'RWalks': ('index.bin', 'RWalks/rwalks/hnswlib/hnswalg.h', 1975),
        'PathSeer': ('index.bin', 'PathSeer/faiss/faiss/impl/index_write.cpp', 855),
        'Curator': ('curator.index', 'thirdparty/curator-v2/3rd_party/faiss/faiss/MultiTenantIndexIVFHierarchical.cpp', 1775),
        'UNG': ('vecs.bin', 'UNG/codes/src/uni_nav_graph_io.cpp', 303),
        'FAVOR': ('index.bin', 'FAVOR/include/favor.h', 336),
        'PTNC': ('vecs.bin', 'UNG/codes/src/uni_nav_graph_io.cpp', 303),
    }
    records = []
    for dataset in DATASETS:
        with (workspace/'FilterVectorData'/dataset/f'{dataset}_base.bin').open('rb') as f:
            n, dim = struct.unpack('<II', f.read(8))
        payload = n*dim*4
        for algorithm, directory in ALGORITHMS.items():
            name, source, line = specs[algorithm]
            path = workspace/'FilterVectorResult'/dataset/'index'/directory/'index_files'/name
            size = path.stat().st_size
            assert size >= payload, f'Vector-containing index is smaller than expected payload: {path}'
            if name == 'vecs.bin':
                with path.open('rb') as f:
                    stored_n, stored_dim = struct.unpack('<II', f.read(8))
                assert (stored_n, stored_dim) == (n, dim), path
                assert size == payload+8, path
            if algorithm == 'FAVOR':
                with path.open('rb') as f:
                    assert f.read(8) == b'FAVROA01', path
            records.append(dict(dataset=dataset, algorithm=algorithm,
                vector_storage='separate_vecs_bin_in_index_directory' if name=='vecs.bin' else 'embedded_in_index_binary',
                vector_file=str(path), vector_file_bytes=size, num_points=n, dim=dim,
                dtype='float32', stored_vector_payload_bytes=payload, stored_vector_copies=1,
                included_vector_bytes=0 if algorithm in ('UNG','PTNC') else payload,
                counted_copies=0 if algorithm in ('UNG','PTNC') else 1,
                persistence_source=f'{project/source}:{line}'))
    write_csv(output/'index_vector_storage_inventory.csv', records)
    print(f'Verified vector-containing artifacts for {len(records)} indexes.')


if __name__ == '__main__':
    verify_vector_storage()
    update_summary(include_vectors=True, excluded_vector_algorithms=('UNG', 'PTNC'))
