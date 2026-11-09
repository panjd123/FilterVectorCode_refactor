"""CPU integration regression for graph-only UNG/Trie block persistence and search.

UNG_GRAPH_ONLY_BIN_DIR points to the apps directory of a compiled build. Optional
UNG_GRAPH_ONLY_REFERENCE_BIN_DIR compares a pre-removal build on the same fixture.
"""
from array import array
import csv
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


AUXILIARY_FILES = (
    'lng_coverage_ratio', 'covered_sets', 'lng_descendants_num',
    'lng_descendants', 'lng_descendants_rb.bin', 'covered_sets_rb.bin',
    'lng_descendants_bits', 'covered_sets_bits', 'trie_label_frequency.csv',
    'vector_attr_graph', 'reordered_vecs.fvecs', 'reordered_labels.txt',
)
CORE_FILES = (
    'vecs.bin', 'labels.txt', 'group_id_to_label_set', 'group_id_to_range',
    'group_entry_points', 'group_id_to_vec_ids.dat', 'new_to_old_vec_ids',
    # Trie serializes hash tables in noncanonical order; search checks its semantics.
    'graph', 'global_graph', 'global_vamana_entry_point',
    'lng_out_neighbors.dat',
)


@unittest.skipUnless(os.environ.get('UNG_GRAPH_ONLY_BIN_DIR'),
                     'requires compiled UNG applications')
class GraphOnlyIndexTest(unittest.TestCase):
    def test_build_load_search_without_auxiliary_caches(self):
        binary = Path(os.environ['UNG_GRAPH_ONLY_BIN_DIR']).resolve()
        reference = os.environ.get('UNG_GRAPH_ONLY_REFERENCE_BIN_DIR')
        reference = Path(reference).resolve() if reference else None
        with tempfile.TemporaryDirectory(prefix='ung_graph_only_') as tmp:
            root = Path(tmp)
            labels = [(1,), (1, 2), (1, 3), (1, 2, 3), (2,), (2, 3),
                      (2, 4), (2, 3, 4), (3,), (3, 4), (4,), (1, 2, 3, 4)]
            rng = random.Random(20261019)
            vectors = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(288)]
            base_labels = [labels[i % len(labels)] for i in range(288)]
            queries = [vectors[i * 17 % 288] for i in range(24)]
            query_labels = [labels[i * 5 % len(labels)] for i in range(24)]
            for name, vecs, lbls in [('base', vectors, base_labels),
                                     ('query', queries, query_labels)]:
                with (root / f'{name}.bin').open('wb') as out:
                    out.write(struct.pack('<II', len(vecs), 16))
                    array('f', (x for row in vecs for x in row)).tofile(out)
                (root / f'{name}.txt').write_text(''.join(
                    ','.join(map(str, row)) + '\n' for row in lbls))
            with (root / 'gt.bin').open('wb') as out:
                for q, ql in zip(queries, query_labels):
                    expected = sorted((sum((a-b)**2 for a, b in zip(q, row)), i)
                                      for i, row in enumerate(vectors)
                                      if set(ql).issubset(base_labels[i]))[:5]
                    for distance, point_id in expected:
                        out.write(struct.pack('<If', point_id, distance))
            (root / 'groups.txt').write_text('0\n' * len(queries))
            env = {k: v for k, v in os.environ.items() if not k.startswith('UNG_')}
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       UNG_BUILD_PROFILE='custom', UNG_GROUP_GRAPH_IMPL='0',
                       UNG_CROSS_EDGE_IMPL='0', UNG_ADDITIONAL_EDGES_IMPL='0',
                       UNG_GROUP_GRAPH_COMPLETE_NX='32', UNG_TRIE_CROSS_SOURCE_EXACT='1',
                       UNG_SKIP_REORDERED_EXPORT='1', UNG_SPECIAL_BLOCK_GPU_INTRA='0',
                       UNG_SPECIAL_BLOCK_GPU_INTER='0', UNG_SPECIAL_INTRA_COMPLETE_NX='128',
                       UNG_VALIDATE_FILTER_RESULTS='1')

            def run(cmd, log, custom_env):
                proc = subprocess.run(list(map(str, cmd)), env=custom_env,
                                      text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
                (root / log).write_text(proc.stdout)
                self.assertEqual(proc.returncode, 0, proc.stdout[-5000:])
                return proc.stdout

            def build(bin_dir, out, topology, old=False):
                out.mkdir(exist_ok=True)
                local_env = dict(env, UNG_BASE_GROUP_TOPOLOGY=topology)
                if old:
                    local_env.update(UNG_BUILD_SEARCH_AUXILIARY='1',
                                     UNG_DESCENDANTS_IMPL='1', UNG_COVERAGE_IMPL='0')
                run([bin_dir / 'build_UNG_index', '--dataset', 'Genome',
                     '--data_type', 'float', '--dist_fn', 'L2',
                     '--base_bin_file', root / 'base.bin',
                     '--base_label_file', root / 'base.txt',
                     '--base_label_info_file', root / 'unused',
                     '--base_label_tree_roots', root / 'unused', '--num_threads', 1,
                     '--index_path_prefix', str(out) + '/',
                     '--result_path_prefix', str(out) + '/', '--max_degree', 16,
                     '--Lbuild', 64, '--num_cross_edges', 2], out.name+'.log', local_env)
                return out

            def blocks(bin_dir, index, out):
                out.mkdir()
                run([bin_dir / 'build_special_block_index',
                     '--ung_index_path_prefix', str(index)+'/',
                     '--block_index_path_prefix', str(out)+'/',
                     '--result_path_prefix', str(out)+'/', '--data_type', 'float',
                     '--dist_fn', 'L2', '--num_threads', 1, '--min_points', 32,
                     '--max_degree', 16, '--num_cross_edges', 2, '--Lbuild', 64],
                    out.name+'.log', dict(env, UNG_BASE_GROUP_TOPOLOGY='trie',
                                         UNG_SPECIAL_BLOCK_DATA_MODE='x1'))
                return out

            def search(bin_dir, index, tag, strategy, sidecar=None, mode='free_state', algorithm=1):
                out = root / tag
                out.mkdir()
                args = [bin_dir / 'search_UNG_index', '--dataset', 'Genome',
                        '--data_type', 'float', '--dist_fn', 'L2',
                        '--base_bin_file', root / 'base.bin',
                        '--query_bin_file', root / 'query.bin',
                        '--query_label_file', root / 'query.txt',
                        '--query_group_id_file', root / 'groups.txt',
                        '--gt_file', root / 'gt.bin', '--K', 5, '--num_threads', 1,
                        '--index_path_prefix', str(index)+'/',
                        '--result_path_prefix', str(out)+'/',
                        '--selector_model_prefix', root / 'unused',
                        '--num_entry_points', 4, '--Lsearch', 64,
                        '--is_new_method', 'true', '--is_idea2_available', 'false',
                        '--is_new_trie_method', 'false', '--is_rec_more_start', 'false',
                        '--skip_query_features', 'true' if bin_dir == reference else 'false',
                        '--skip_bitmap_comparison', 'true' if bin_dir == reference else 'false',
                        '--force_use_alg', algorithm, '--lsearch_start', 0, '--lsearch_step', 0,
                        '--efs_start', 0, '--efs_step_slow', 0, '--efs_step_fast', 0,
                        '--lsearch_threshold', 0, '--entry_group_strategy', strategy]
                if sidecar:
                    args += ['--block_index_path_prefix', str(sidecar)+'/']
                if algorithm in (0, 3, 4):
                    proc = subprocess.run(list(map(str, args)), env=env, text=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn('routes were removed', proc.stdout)
                    return
                # Load the reference caches to verify their removal does not affect graph work.
                text = run(args, tag+'.log', dict(env, UNG_DISABLE_ENTRY_ROUTE_STATS='0',
                           UNG_SPECIAL_BLOCK_SEARCH='1' if sidecar else '0',
                           UNG_SPECIAL_SEARCH_MODE=mode))
                self.assertIn('violations=0', text)
                self.assertFalse((out / 'query_features.csv').exists())
                with (out / 'query_details_repeat1.csv').open() as detail_file:
                    details = list(csv.DictReader(detail_file))
                return details

            report = {}
            for topology, strategy in [('lng', 'original'), ('trie', 'trie')]:
                index = build(binary, root / topology, topology)
                for name in AUXILIARY_FILES:
                    self.assertFalse((index / name).exists(), name)
                sidecar = blocks(binary, index, root / 'blocks') if topology == 'trie' else None
                modes = ['free_state', 'favor_blocks'] if sidecar else ['free_state']
                old_index = build(reference, root / ('old_'+topology), topology, True) if reference else None
                old_blocks = blocks(reference, old_index, root/'old_blocks') if reference and sidecar else None
                if reference:
                    for name in CORE_FILES:
                        self.assertEqual((index/name).read_bytes(), (old_index/name).read_bytes(), name)
                for mode in modes:
                    current = search(binary, index, topology+'_'+mode, strategy, sidecar, mode)
                    # Ignored corrupt old files must never be parsed by the new loader.
                    for name in AUXILIARY_FILES:
                        (index / name).write_bytes(b'obsolete cache: deliberately invalid')
                    ignored = search(binary, index, topology+'_'+mode+'_ignored', strategy, sidecar, mode)
                    columns = ['Recall', 'DistCalcs', 'NumNodeVisited', 'TotalDistanceCalcs',
                               'RegularEdgesScanned', 'FreeEdgesScanned', 'NumEntries']
                    # Header names differ across older experiment revisions.
                    columns = [k for k in current[0] if k in columns or
                               k.lower() in ('recall', 'numdistancecalcs', 'numnodesvisited',
                                             'num_distance_calcs', 'num_nodes_visited')]
                    self.assertTrue(columns, list(current[0]))
                    projection = lambda rows: [[row[k] for k in columns] for row in rows]
                    self.assertEqual(projection(current), projection(ignored))
                    if reference:
                        previous = search(reference, old_index, topology+'_'+mode+'_old', strategy, old_blocks, mode)
                        self.assertEqual(projection(current), projection(previous))
                        compatible = search(binary, old_index, topology+'_'+mode+'_legacy_index',
                                            strategy, old_blocks, mode)
                        self.assertEqual(projection(current), projection(compatible))
                    report[topology+'_'+mode] = {
                        'queries': len(current), 'compared_columns': columns,
                        'mean_recall': sum(float(row['Recall']) for row in current) / len(current)}
                for algorithm in (0, 3, 4):
                    search(binary, index, topology+'_removed_'+str(algorithm), strategy,
                           algorithm=algorithm)
                # Rebuilding an existing prefix removes obsolete files as well.
                build(binary, index, topology)
                for name in AUXILIARY_FILES:
                    self.assertFalse((index / name).exists(), name)
            output = os.environ.get('UNG_GRAPH_ONLY_REPORT')
            if output:
                Path(output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
