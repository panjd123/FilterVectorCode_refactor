import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gpu_naive_runner_test', HERE/'run_gpu_naive.py')
naive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(naive)

class NaiveContractTests(unittest.TestCase):
    def test_only_requested_baseline_factors_differ(self):
        settings = naive.experiment.load_settings(HERE/'config_gpu_naive_eight.json')
        configs = naive.experiment.load_configs(settings, settings['datasets'], settings['variants'])
        self.assertEqual(len(configs), 8)
        for (dataset, name), config in configs.items():
            self.assertEqual(name, 'gpu_naive')
            original = json.loads((HERE/'configs'/dataset/'gpu.json').read_text())
            self.assertNotEqual(config['build_dir'], original['build_dir'])
            self.assertEqual(config['build'], original['build'])
            changes = {key:(original['ung_env'].get(key),value) for key,value in config['ung_env'].items()
                       if original['ung_env'].get(key) != value}
            self.assertEqual(changes, {'UNG_TAGORE_NAIVE':(None,'1'), 'UNG_TAGORE_COMPACT_D2H':('1','0')})

    def test_dispatch_metadata_and_full_rows_are_required(self):
        config=json.loads((HERE/'configs_gpu_naive/Amazon/gpu_naive.json').read_text())
        meta={'gpu_intra_naive':'1','gpu_intra_distance_kernel':'cuda_core','gpu_intra_compact_d2h':'0',
              'gpu_intra_blocks':'1','gpu_intra_points':'501'}
        trace='[TagoreCuda][naive] distance_kernel=cuda_core points=501 k=96 graph_stride=96 streams=1 optimized_topk=0 compact_d2h=0\n'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); case=root/'case'; case.mkdir()
            (case/'config.json').write_text(json.dumps(config))
            (case/'build.log').write_text(trace)
            with mock.patch.object(naive,'_original_collect',return_value=({},{})), \
                 mock.patch.object(naive.experiment.common,'read_meta',return_value=meta):
                self.assertEqual(naive.collect({'case':'case'},root), ({},{}))
                for bad in (trace.replace('graph_stride=96','graph_stride=33'),
                            trace.replace('points=501','points=500'), ''):
                    (case/'build.log').write_text(bad)
                    with self.assertRaises(ValueError): naive.collect({'case':'case'},root)
                (case/'build.log').write_text(trace)
                meta['gpu_intra_distance_kernel']='wmma'
                with self.assertRaises(ValueError): naive.collect({'case':'case'},root)

if __name__=='__main__': unittest.main()
