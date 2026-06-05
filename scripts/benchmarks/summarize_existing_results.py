#!/usr/bin/env python3
import csv
from pathlib import Path


def read_build_time(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as f:
        return dict(csv.reader(f))


def main() -> None:
    rows: list[dict[str, str]] = []

    celeba = Path("/home/graphdb/FilterVectorCode_old_rerun_results/celeba_c2_20260519_111420/results/build_time.csv")
    bt = read_build_time(celeba)
    if bt:
        rows.append({
            "section": "celeba_cpu_original",
            "case": "old_cpu_rerun",
            "index_ms": bt.get("index_time", ""),
            "graph_ms": bt.get("build_graph_time", ""),
            "lng_ms": bt.get("build_LNG_time", ""),
            "desc_ms": bt.get("cal_descendants_time", ""),
            "coverage_ms": bt.get("cal_coverage_ratio_time", ""),
            "cross_edges_ms": bt.get("build_cross_edges_time", ""),
            "source": str(celeba),
        })

    for label, path in [
        ("sift30_cpu_cross_edge", Path("/home/graphdb/FilterVectorResultsRefactor/sift30_zipf_origstyle_current_cpu_20260519_133655/results/build_time.csv")),
        ("sift30_gpu_cross_edge", Path("/home/graphdb/FilterVectorResultsRefactor/sift30_gpu_default_heavy_sgemm_20260519_135430/results/build_time.csv")),
    ]:
        bt = read_build_time(path)
        if bt:
            rows.append({
                "section": label,
                "case": path.parent.parent.name,
                "index_ms": bt.get("index_time", ""),
                "graph_ms": bt.get("build_graph_time", ""),
                "lng_ms": bt.get("build_LNG_time", ""),
                "desc_ms": bt.get("cal_descendants_time", ""),
                "coverage_ms": bt.get("cal_coverage_ratio_time", ""),
                "cross_edges_ms": bt.get("build_cross_edges_time", ""),
                "source": str(path),
            })

    fused_root = Path("/home/graphdb/FilterVectorResultsRefactor/fused_topk_sift30_20260519_1530")
    for case in ["nx256_511_w8", "nx512_1023_w8", "nx1024_4096_w8", "verify_nx256_511", "debug_launch_nx256_511_v2"]:
        path = fused_root / case / "results/build_time.csv"
        bt = read_build_time(path)
        if bt:
            prof = fused_root / case / "others/ung_prof.log"
            rows.append({
                "section": "fused_topk_sift30",
                "case": case,
                "index_ms": bt.get("index_time", ""),
                "graph_ms": bt.get("build_graph_time", ""),
                "lng_ms": bt.get("build_LNG_time", ""),
                "desc_ms": bt.get("cal_descendants_time", ""),
                "coverage_ms": bt.get("cal_coverage_ratio_time", ""),
                "cross_edges_ms": bt.get("build_cross_edges_time", ""),
                "source": f"{path}; {prof}",
            })

    tagore_csv = Path("/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/tagore_all_large_build_times.csv")
    if tagore_csv.exists():
        vals = []
        with tagore_csv.open() as f:
            for row in csv.DictReader(f):
                vals.append(float(row["total_s_wall"]) * 1000.0)
        vals_sorted = sorted(vals)
        if vals:
            rows.append({
                "section": "tagore_build_nx_ge_1024",
                "case": "119_groups",
                "index_ms": f"{sum(vals):.3f}",
                "graph_ms": f"{sum(vals):.3f}",
                "lng_ms": "",
                "desc_ms": "",
                "coverage_ms": "",
                "cross_edges_ms": "",
                "source": str(tagore_csv),
            })

    fields = ["section", "case", "index_ms", "graph_ms", "lng_ms", "desc_ms", "coverage_ms", "cross_edges_ms", "source"]
    print(",".join(fields))
    for row in rows:
        print(",".join('"' + row.get(f, "").replace('"', '""') + '"' for f in fields))


if __name__ == "__main__":
    main()
