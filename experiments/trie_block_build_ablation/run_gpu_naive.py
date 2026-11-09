#!/usr/bin/env python3
"""Isolated eight-dataset GPU-Naive experiment; existing cohorts are untouched.

Whole-index timing exception matches run.py: payload=1, process repeats, no L2
bypass and no CUDA Graph capture. Every timed sample uses gpulock perf on GPU0.
Sampling, iteration counts and FastGrnnd budgets match the existing GPU cases.
Only local-join distance computation, stream count, candidate update selection,
and graph D2H layout define this baseline. This is not a recall-equivalence test.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("naive_build_experiment", HERE / "run.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
_original_collect = experiment.collect

def collect(step, root, identity=None):
    row, observed = _original_collect(step, root, identity)
    case = root / step["case"]
    config = json.loads((case / "config.json").read_text())
    env = config["ung_env"]
    expected = {"UNG_TAGORE_NAIVE": "1", "UNG_TAGORE_BATCH_STREAMS": "1",
                "UNG_TAGORE_OPTIMIZED_TOPK": "0", "UNG_TAGORE_COMPACT_D2H": "0"}
    if any(env.get(k) != v for k,v in expected.items()):
        raise ValueError("GPU-Naive contract changed")
    meta = experiment.common.read_meta(case / "block_index_files/meta")
    if any(meta.get(k) != v for k,v in {
        "gpu_intra_naive": "1", "gpu_intra_distance_kernel": "cuda_core",
        "gpu_intra_compact_d2h": "0"}.items()):
        raise ValueError("Naive kernel metadata is missing")
    log = (case / "build.log").read_text()
    traces = re.findall(r"\[TagoreCuda\]\[naive\] distance_kernel=cuda_core points=(\d+) k=(\d+) graph_stride=(\d+) streams=1 optimized_topk=0 compact_d2h=0", log)
    if len(traces) != int(meta["gpu_intra_blocks"]):
        raise ValueError("Naive kernel was not dispatched for every GPU block")
    if sum(int(t[0]) for t in traces) != int(meta["gpu_intra_points"]):
        raise ValueError("Naive trace point coverage differs from metadata")
    if any(int(k) != int(stride) or int(k) != 96 for _,k,stride in traces):
        raise ValueError("GPU-Naive did not return complete 96-slot graph rows")
    return row, observed

experiment.collect = collect

REFERENCE = HERE.parents[1] / "tools/pic-draw/trie_block_build_optimizations/current_four_methods_wide.csv"

def progress(root, manifest):
    # Reports are generated between samples, outside the gpulock timing worker.
    snapshot = root / "reference_four_methods.csv"
    if not snapshot.exists():
        snapshot.write_bytes(REFERENCE.read_bytes())
        (root / "reference_provenance.json").write_text(json.dumps({
            "source": str(REFERENCE),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "relationship": "historical reference only; separate run and different builder hash; one sample per method",
            "note": "naive changes local-join distance arithmetic and D2H layout; approximate graph quality is not established"
        }, indent=2) + "\n")
    with snapshot.open() as stream:
        reference = {r["dataset"]: r for r in csv.DictReader(stream)}
    rows = []
    summary = root / "comparison.csv"
    if summary.exists():
        with summary.open() as stream:
            for row in csv.DictReader(stream):
                old = reference[row["dataset"]]
                item = {"dataset": row["dataset"], "samples": row["samples"],
                    "naive_intra_s": float(row["intra_ms_median"]) / 1000,
                    "naive_end_to_end_s": float(row["end_to_end_wall_ms_median"]) / 1000}
                for method in ("gpu", "gpu_balance_topk"):
                    for metric in ("intra", "end_to_end"):
                        item[method + "_" + metric + "_s"] = float(old[method + "_" + metric + "_s"])
                item["naive_over_gpu_intra"] = item["naive_intra_s"] / item["gpu_intra_s"]
                item["naive_over_full_intra"] = item["naive_intra_s"] / item["gpu_balance_topk_intra_s"]
                rows.append(item)
    experiment.common.write_csv(root / "naive_vs_existing.csv", rows)
    text = ["# GPU-Naive 与现有 GPU 结果的独立参考对比", "",
        "新增测量不覆盖旧四组结果。已有 GPU/完整方案来自 reference_four_methods.csv 快照。",
        "跨批次且构建器哈希不同，每方式每数据集一次、无独立预热；本表不证明等召回率或稳定加速。",
        "Naive = CUDA Core local join + 单 stream + 完整候选排序 + 96 槽位完整回传；其他构建参数相同。", "",
        "| 数据集 | Naive 块内/s | 旧 GPU 块内/s | 完整方案块内/s | Naive 端到端/s | 旧 GPU 端到端/s | 完整方案端到端/s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    by_dataset = {r["dataset"]: r for r in rows}
    for dataset in manifest["datasets"]:
        row = by_dataset.get(dataset)
        keys = ("naive_intra_s", "gpu_intra_s", "gpu_balance_topk_intra_s",
                "naive_end_to_end_s", "gpu_end_to_end_s", "gpu_balance_topk_end_to_end_s")
        cells = [f"{row[k]:.3f}" for k in keys] if row else ["—"] * len(keys)
        text.append("| " + dataset + " | " + " | ".join(cells) + " |")
    text += ["", f"完成 {len(rows)}/{len(manifest['datasets'])} 个数据集。详细阶段及审计信息见 comparison.md、manifest.json 和各 case 的日志。", ""]
    (root / "naive_vs_existing.md").write_text("\n".join(text))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--config" not in argv:
        argv = ["--config", str(HERE / "config_gpu_naive_eight.json"), *argv]
    raise SystemExit(experiment.main(argv, on_progress=progress))
