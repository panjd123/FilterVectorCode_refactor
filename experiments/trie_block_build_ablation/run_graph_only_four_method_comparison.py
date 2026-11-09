#!/usr/bin/env python3
"""Rerun the first three graph-only methods and automatically publish all four.

Timing stays in run.py/common's gpulock worker. Progress reports and final plots
run synchronously between samples, outside every timed interval. Builders and
the timing worker are copied into an immutable per-run runtime directory.
"""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULT_ROOT = REPO.parent/'FilterVectorResult/trie_block_build_ablation'
DEFAULT_REFERENCE = RESULT_ROOT/'20261019_113719_771827_gpu_balance_topk_graph_only'
PLOT_DIR = REPO/'tools/pic-draw'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify_reference(experiment, reference, settings, configs):
    previous = json.loads((reference/'manifest.json').read_text())
    if previous['status']!='complete':
        raise ValueError('The fourth-method reference run is incomplete')
    if experiment.digest(experiment.COMMON_PATH)!=previous['timing_worker_sha256']:
        raise ValueError('Timing worker changed since the fourth method')
    if len(previous['steps'])!=8 or any(step['variant']['name']!='gpu_balance_topk' for step in previous['steps']):
        raise ValueError('Expected eight completed fourth-method samples')
    build_dir = Path(next(iter(configs.values()))['build_dir'])
    for name,digest in previous['binary_sha256'].items():
        if experiment.digest(build_dir/'apps'/name)!=digest:
            raise ValueError(f'Builder changed since the fourth method: {name}')
    for dataset in settings['datasets']:
        step = next(step for step in previous['steps'] if step['dataset']==dataset)
        previous_config = json.loads((reference/step['case']/'config.json').read_text())
        for variant in settings['variants']:
            config = configs[dataset,variant['name']]
            if experiment.input_info(config)!=previous['inputs'][dataset]:
                raise ValueError(f'Input dataset changed: {dataset}')
            if config['build']!=previous_config['build']:
                raise ValueError(f'Common build parameters changed: {dataset}')
            a,b = config['ung_env'],previous_config['ung_env']
            factors = {'UNG_SPECIAL_BLOCK_GPU_INTRA','UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT',
                       'UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED','UNG_TAGORE_BATCH_STREAMS',
                       'UNG_TAGORE_OPTIMIZED_TOPK'}
            if {k:v for k,v in a.items() if k not in factors}!={k:v for k,v in b.items() if k not in factors}:
                raise ValueError(f'Other build settings changed: {dataset}')
    return previous


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--run-root',type=Path)
    args=parser.parse_args(argv)
    experiment=module('current_build_experiment',HERE/'run.py')
    plot=module('current_build_plot',PLOT_DIR/'plot_trie_block_build_optimizations.py')
    profile=HERE/'config_first_three_eight.json'
    settings=experiment.load_settings(profile)
    configs=experiment.load_configs(settings,settings['datasets'],settings['variants'])
    reference=args.reference_run.resolve()
    previous=verify_reference(experiment,reference,settings,configs)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    root=(args.run_root or RESULT_ROOT/(stamp+'_graph_only_first_three')).resolve()
    if root.exists():
        raise FileExistsError(root)
    runtime=root.with_name(root.name+'_runtime')
    runtime.mkdir(parents=True,exist_ok=False)
    (runtime/'apps').mkdir()
    build_dir=Path(next(iter(configs.values()))['build_dir'])
    for name,digest in previous['binary_sha256'].items():
        shutil.copy2(build_dir/'apps'/name,runtime/'apps'/name)
        if experiment.digest(runtime/'apps'/name)!=digest:
            raise ValueError('Builder snapshot verification failed')
    worker=runtime/'timing_worker.py'
    shutil.copy2(experiment.COMMON_PATH,worker)
    experiment.COMMON_PATH=worker
    runtime_settings=json.loads(profile.read_text())
    runtime_settings['source_config_dir']=settings['source_config_dir']
    runtime_settings['config_dir']=str(runtime/'configs')
    for (dataset,name),config in configs.items():
        config=json.loads(json.dumps(config))
        config['build_dir']=str(runtime)
        path=runtime/'configs'/dataset/(name+'.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(config,indent=2)+'\n')
    runtime_profile=runtime/'run_config.json'
    runtime_profile.write_text(json.dumps(runtime_settings,indent=2)+'\n')
    config_path=PLOT_DIR/'trie_block_build_optimizations_config.json'
    plot_config=json.loads(config_path.read_text())
    plot_config['graph_only_sources']=[
        {'run_dir':str(reference),'variant_mapping':{'gpu_balance_topk':'gpu_balance_topk'}},
        {'run_dir':str(root),'variant_mapping':{v['name']:v['name'] for v in settings['variants']}}]
    frozen_plot_config=runtime/'plot_sources.json'
    frozen_plot_config.write_text(json.dumps(plot_config,indent=2)+'\n')
    reference_samples,_=plot.load_cohort(plot_config['graph_only_sources'][:1],REPO.parent/'FilterVectorResult',
                                       'graph_only',plot_config['dataset_order'])
    data_dir=PLOT_DIR/'trie_block_build_optimizations'
    pictures=PLOT_DIR/'pic/trie_block_build_optimizations'
    data_dir.mkdir(parents=True,exist_ok=True)
    if (data_dir/'summary.md').exists() and not (data_dir/'historical_summary.md').exists():
        shutil.copy2(data_dir/'summary.md',data_dir/'historical_summary.md')
    if pictures.exists() and not (pictures/'historical').exists():
        (pictures/'historical').mkdir()
        for path in list(pictures.glob('*.png'))+list(pictures.glob('*.pdf')):
            shutil.copy2(path,pictures/'historical'/path.name)
    config_path.write_text(json.dumps(plot_config,indent=2)+'\n')
    receipt={'run_root':str(root),'runtime':str(runtime),'reference_run':str(reference),
             'binary_sha256':previous['binary_sha256'],'timing_worker_sha256':previous['timing_worker_sha256'],
             'expected_new_samples':24,'reused_fourth_method_samples':8}
    (runtime/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Run root:',root,flush=True)
    print('Immutable runtime:',runtime,flush=True)

    def progress(current_root,manifest):
        if manifest.get('binary_sha256') and manifest['binary_sha256']!=previous['binary_sha256']:
            raise ValueError('Current run and fourth-method builder hashes differ')
        samples=list(reference_samples)
        for step in manifest['steps']:
            if step['status']!='complete':
                continue
            row,_=experiment.collect(step,current_root)
            meta=experiment.common.read_meta(current_root/step['case']/'index_files/meta')
            if meta.get('search_auxiliary_built')!='0':
                raise ValueError('Unexpected auxiliary precomputation in new base')
            samples.append({'cohort':'graph_only','dataset':row['dataset'],'method':row['variant'],
                            **{key:float(row[value])/1000 for key,value in plot.METRICS.items()},
                            'source_run':str(current_root),'case_dir':row['case_dir'],
                            'inter_reserve':1,'audit_status':'case_config_and_meta_verified'})
        rows=plot.aggregate(samples,plot_config['dataset_order'],plot.METHODS,require_complete=False)
        plot.write_csv(data_dir/'current_four_methods.csv',rows)
        plot.write_csv(data_dir/'current_four_methods_wide.csv',plot.wide_table(rows,plot_config['dataset_order'],allow_missing=True))
        plot.write_csv(data_dir/'current_speedups.csv',plot.speedup_table(rows,plot_config['dataset_order'],allow_missing=True))
        for name in ('current_summary.md','summary.md'):
            plot.write_current_summary(data_dir/name,rows,plot_config['dataset_order'],pictures,manifest['status'])
        status={**receipt,'status':manifest['status'],'completed_cells':len(rows),
                'new_complete':sum(s['status']=='complete' for s in manifest['steps']),
                'failures':[{'dataset':s['dataset'],'method':s['variant']['name'],'error':s.get('error')}
                            for s in manifest['steps'] if s['status']=='failed']}
        (data_dir/'current_status.json').write_text(json.dumps(status,indent=2)+'\n')
        if current_root.exists():
            shutil.copy2(data_dir/'current_summary.md',current_root/'four_method_comparison.md')
            shutil.copy2(data_dir/'current_four_methods.csv',current_root/'four_method_comparison.csv')
        print(f"Summary updated: {len(rows)}/32 cells; {status['new_complete']}/24 new samples",flush=True)
        if manifest['status']=='complete':
            plot.main(['--config',str(frozen_plot_config),'--cohort','graph_only'])
            shutil.copy2(data_dir/'current_summary.md',current_root/'four_method_comparison.md')
            status['plots_status']='complete'
            (data_dir/'current_status.json').write_text(json.dumps(status,indent=2)+'\n')

    progress(root,{'status':'running','steps':[]})
    code=experiment.main(['--config',str(runtime_profile),'--run-root',str(root),
                          '--repeat','1','--warmup','0','--skip-build','--discard-index-data'],on_progress=progress)
    return code


if __name__=='__main__':
    raise SystemExit(main())
