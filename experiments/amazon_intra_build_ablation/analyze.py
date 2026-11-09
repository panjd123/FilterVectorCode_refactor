#!/usr/bin/env python3
"""Correct historical timing scope and decompose the combined-method slowdown."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import profile_values, read_meta, summarize, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--base-meta", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    with (root / "all_runs.csv").open(newline="") as stream:
        original = list(csv.DictReader(stream))
    rows = []
    for original_row in original:
        row = dict(original_row)
        for key in ("intra_ms", "inter_ms", "metadata_ms", "save_ms", "block_index_ms"):
            row[key] = float(row[key])
        case = Path(row["case_dir"])
        base_meta = case / "index_files/meta"
        if not base_meta.is_file():
            if not args.base_meta:
                raise ValueError("historical block-only run requires --base-meta")
            base_meta = args.base_meta
        base_ms = float(read_meta(base_meta)["index_time_add_rb(ms)"])
        row["base_build_timer_ms"] = base_ms
        row["total_index_build_ms"] = base_ms + row["block_index_ms"]
        if "end_to_end_wall_ms" in row:
            for key in ("end_to_end_wall_ms", "base_wall_ms", "block_wall_ms"):
                row[key] = float(row[key])
        row["other_ms"] = row["block_index_ms"] - sum(row[key] for key in
                            ("intra_ms", "inter_ms", "metadata_ms", "save_ms"))
        row["base_meta"] = str(base_meta)
        row.update(profile_values((case / "build.log").read_text()))
        rows.append(row)
    summary = summarize(rows)
    # Keep original raw data and original block-only table unchanged.
    write_csv(root / "comparison_with_base.csv", summary)
    text = ["# Amazon 总构建时间与变慢原因", "",
            "总构建计时 = 基础 Trie index_time_add_rb(ms) + 块 build_time(ms)，与现有汇总脚本口径一致。", "",
            "历史数据的相加结果不能冒充从头到尾重新测得的墙钟；真实墙钟只取 timing.json 中的 end_to_end_wall_ms。", "",
            "| 方法 | 基础构建计时/s | 块构建计时/s | 总构建计时/s |", "|---|---:|---:|---:|"]
    for item in summary:
        text.append(f"| {item['variant']} | {item['base_build_timer_ms_median']/1000:.3f} | "
                    f"{item['block_index_ms_median']/1000:.3f} | {item['total_index_build_ms_median']/1000:.3f} |")
    measured = [row for row in rows if row["phase"] == "measured"]
    stages = ("intra_ms", "inter_ms", "metadata_ms", "save_ms", "other_ms", "block_index_ms")
    means = {}
    for variant in ("topk_only", "load_balance_topk"):
        group = [row for row in measured if row["variant"] == variant]
        if group:
            means[variant] = {key: statistics.mean(row[key] for row in group) for key in stages}
    if len(means) == 2:
        text += ["", "阶段均值分解：组合方案减去单 stream topK。用均值保证各阶段差值可以相加。", "",
                 "| 阶段 | 单 stream topK/s | 组合/s | 差值/s |", "|---|---:|---:|---:|"]
        for key in stages:
            before, after = means["topk_only"][key], means["load_balance_topk"][key]
            text.append(f"| {key} | {before/1000:.3f} | {after/1000:.3f} | {(after-before)/1000:+.3f} |")
        counted = {}
        for variant in means:
            group = [row for row in measured if row["variant"] == variant and row.get("graph_distance_computations")]
            if group:
                counted[variant] = {
                    "distance_computations": statistics.mean(int(row["graph_distance_computations"]) for row in group),
                    "inter_ms": statistics.mean(float(row["inter_ms"]) for row in group),
                    "end_to_end_wall_ms": statistics.mean(float(row["end_to_end_wall_ms"]) for row in group),
                }
        if len(counted) == 2:
            text += ["", "实际工作量（本轮计数，不能回填到历史试验）：", "",
                     "| 方法 | 块间距离计算次数 | 块间/s | 真实端到端/s |", "|---|---:|---:|---:|"]
            for variant, values in counted.items():
                text.append(f"| {variant} | {values['distance_computations']:,.0f} | "
                            f"{values['inter_ms']/1000:.3f} | {values['end_to_end_wall_ms']/1000:.3f} |")
            plain, combined = counted["topk_only"], counted["load_balance_topk"]
            work_delta = (combined["distance_computations"] / plain["distance_computations"] - 1) * 100
            wall_delta = (combined["inter_ms"] / plain["inter_ms"] - 1) * 100
            text += ["", f"本轮块间距离计算量只增加 {work_delta:.4f}%，块间墙钟却增加 {wall_delta:.2f}%。"
                     "额外距离计算不足以解释时间差，变慢定位在 CPU 阶段的执行效率；"
                     "还需隔离缓存局部性、内存分配与缓存池锁等待，才能确认具体根因。"]
    text += ["", "已知与限制：", "",
             "- 块间阶段使用 CPU 图搜索，并没有使用块内的多 stream 调度；任务数相同不代表实际距离计算次数相同。",
             "- GPU 初始化使用 clock() 随机种子，反向候选使用原子追加；不同运行或调度可能产生不同局部图，影响后续 CPU 搜索。",
             "- 即使距离计算次数接近，访问的向量、图的缓存局部性和内存分配状态仍可能不同；计数不能单独证明缓存行为相同。",
             "- 块间每次查询都通过共享 SearchCacheList 的 mutex 申请/归还缓存，多线程下存在锁竞争；非自愿上下文切换只能作为调度压力线索，不能单独证明某个锁就是根因。",
             "- 保存、加载等无关阶段也会波动；只有一次样本时，不能断言是组合算法退化，更不能断言召回率下降。",
             "- 新版日志记录 graph_distance_computations；结合重复运行和反向顺序，区分搜索工作量变化与调度/系统波动。", ""]
    (root / "analysis.md").write_text("\n".join(text))
    (root / "analysis.json").write_text(json.dumps({"rows": rows, "stage_means": means}, indent=2) + "\n")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
