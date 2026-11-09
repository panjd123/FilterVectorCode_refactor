"""Check sample selection, provenance boundaries, units, and speedup direction."""
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1]/'pic-draw/plot_trie_block_build_optimizations.py'
SPEC = importlib.util.spec_from_file_location('build_optimization_plot', SCRIPT)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class BuildOptimizationSummaryTest(unittest.TestCase):
    def fixture_run(self, root, name='run', binary='historical'):
        run = root/name
        run.mkdir()
        manifest = {'status':'complete', 'settings':{'inter_reserve':True},
                    'binary_sha256':{'builder':binary}, 'steps':[]}
        (run/'manifest.json').write_text(json.dumps(manifest))
        rows = []
        for i, (phase, milliseconds) in enumerate([('warmup',999999),('measured',1000),
                                                    ('measured',10000),('measured',4000)]):
            row = {'dataset':'Genome','variant':'gpu','phase':phase,'repeat':i,
                   'gpu_intra':1,'load_balance':0,'optimized_topk':0,
                   **{metric:milliseconds for metric in plot.METRICS.values()},
                   'case_dir':str(run/f'archived_{i}')}
            row['block_wall_ms'] = milliseconds*2
            row['end_to_end_wall_ms'] = milliseconds*3
            rows.append(row)
        with (run/'all_runs.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        measured = [row for row in rows if row['phase']=='measured']
        summary = {'dataset':'Genome','variant':'gpu','samples':3, **{
            key+'_median':statistics.median(row[key] for row in measured)
            for key in plot.METRICS.values()}}
        with (run/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary));writer.writeheader();writer.writerow(summary)
        return {'run_dir':name,'variant_mapping':{'gpu':'gpu'}}

    def test_excludes_warmup_converts_ms_and_tolerates_explicitly_archived_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture_run(root)
            rows,audit=plot.load_cohort([source],root,'historical',['Genome'])
            self.assertEqual(len(rows),3)
            output=plot.aggregate(rows,['Genome'],('gpu',))[0]
            self.assertEqual(output['samples'],3)
            self.assertEqual(output['intra_s'],4)
            self.assertEqual(output['end_to_end_s'],12)
            self.assertEqual(output['end_to_end_s_min'],3)
            self.assertEqual(output['end_to_end_s_max'],30)
            self.assertEqual(output['audit_status'],'raw_csv_and_manifest_only')
            self.assertIsNone(audit[0]['included_cases'][0]['config_sha256'])
            self.assertTrue(audit[0]['comparison_sha256'])

    def test_rejects_different_builders_within_a_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=self.fixture_run(root,'a','old')
            b=self.fixture_run(root,'b','new')
            with self.assertRaisesRegex(ValueError,'Different builders'):
                plot.load_cohort([a,b],root,'historical',['Genome'])

    def test_cached_summary_cannot_silently_replace_raw_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture_run(root)
            p=root/'run/comparison.csv'
            with p.open(newline='') as stream:rows=list(csv.DictReader(stream))
            rows[0]['end_to_end_wall_ms_median']='99999'
            with p.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Summary disagrees'):
                plot.load_cohort([source],root,'historical',['Genome'])

    def test_graph_only_requires_its_case_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture_run(root)
            with self.assertRaisesRegex(ValueError,'Missing current case files'):
                plot.load_cohort([source],root,'graph_only',['Genome'])

    def test_rejects_a_cross_version_or_incomplete_four_method_matrix(self):
        rows=[{'cohort':'historical','dataset':'Genome','method':'gpu'},
              {'cohort':'graph_only','dataset':'Genome','method':'gpu_balance_topk'}]
        with self.assertRaisesRegex(ValueError,'Cannot mix'):
            plot.aggregate(rows,['Genome'],plot.METHODS)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            plot.aggregate(rows[:1],['Genome'],plot.METHODS)

    def test_pending_cells_are_empty_and_do_not_create_speedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture_run(root)
            samples,_=plot.load_cohort([source],root,'historical',['Genome'])
            rows=plot.aggregate(samples,['Genome'],plot.METHODS,require_complete=False)
            self.assertEqual(len(rows),1)
            wide=plot.wide_table(rows,['Genome'],allow_missing=True)[0]
            self.assertEqual(wide['gpu_end_to_end_s'],12)
            self.assertEqual(wide['cpu_end_to_end_s'],'')
            self.assertEqual(wide['gpu_balance_topk_samples'],'')
            effects=plot.speedup_table(rows,['Genome'],allow_missing=True)[0]
            self.assertEqual(effects['end_to_end_s_cpu_to_gpu_speedup'],'')

    def test_completed_current_matrix_publishes_current_data_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config_path=root/'sources.json';config_path.write_text('{}')
            samples=[]
            for method in plot.METHODS:
                samples.append({'cohort':'graph_only','dataset':'Genome','method':method,
                                **{key:1.0 for key in plot.METRICS},'source_run':'same_version',
                                'case_dir':method,'inter_reserve':1,'audit_status':'verified'})
            args=SimpleNamespace(config=config_path,result_root=root,data_dir=root/'data',output_dir=root/'plots')
            with mock.patch.object(plot,'load_cohort',return_value=(samples,[])), \
                 mock.patch.object(plot,'draw_time_chart') as draw_time, \
                 mock.patch.object(plot,'draw_gpu_speedups') as draw_speed:
                self.assertEqual(plot.render_graph_only(args,{'graph_only_sources':[]},['Genome']),0)
            self.assertEqual(draw_time.call_count,2)
            draw_speed.assert_called_once()
            text=(args.data_dir/'summary.md').read_text()
            self.assertIn('4/4',text)
            self.assertIn('新版 Trie block',text)
            self.assertTrue((args.data_dir/'current_source_manifest.json').exists())
            with (args.data_dir/'current_four_methods.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual({r['cohort'] for r in rows},{'graph_only'})
            self.assertEqual(len(rows),4)

    def test_reports_block_speedup_separately_from_end_to_end_regression(self):
        rows=[{'dataset':'Genome','method':method,'end_to_end_s':e2e,'intra_s':intra}
              for method,e2e,intra in zip(plot.METHODS,[100,20,18,19],[80,10,8,6])]
        output=plot.speedup_table(rows,['Genome'])[0]
        self.assertEqual(output['end_to_end_s_cpu_to_gpu_speedup'],5)
        self.assertAlmostEqual(output['intra_s_lb_to_lb_topk_speedup'],8/6)
        self.assertAlmostEqual(output['intra_s_lb_to_lb_topk_reduction_pct'],25)
        self.assertLess(output['end_to_end_s_lb_to_lb_topk_speedup'],1)
        self.assertLess(output['end_to_end_s_lb_to_lb_topk_reduction_pct'],0)


if __name__=='__main__':
    unittest.main()
