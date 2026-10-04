#!/usr/bin/env python3
"""Summarize the complete L/T base-by-overlay topology factorial.

The input CSV files are produced by ``experiment_cli.py summarize``.  This
script treats the campaign as a screen: ratios are descriptive and are not
reported as confidence intervals because the two halves were not paired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METHOD_TO_CODE = {
    "l0_lng_entry_optimized_lng": "L",
    "l0_trie_entry_trie": "T",
    "l1_t1024_lng_entry_optimized_lng": "LL",
    "l1_t1024_trie_entry_optimized_lng": "LT",
    "l2_t1024_16384_ll_entry_optimized_lng": "LLL",
    "l2_t1024_16384_lt_entry_optimized_lng": "LLT",
    "l2_t1024_16384_tl_entry_optimized_lng": "LTL",
    "l2_t1024_16384_tt_entry_optimized_lng": "LTT",
    "l1_base_trie_t1024_lng_entry_trie": "TL",
    "l1_base_trie_t1024_trie_entry_trie": "TT",
    "l2_base_trie_t1024_16384_ll_entry_trie": "TLL",
    "l2_base_trie_t1024_16384_lt_entry_trie": "TLT",
    "l2_base_trie_t1024_16384_tl_entry_trie": "TTL",
    "l2_base_trie_t1024_16384_tt_entry_trie": "TTT",
}
WORKLOAD_ORDER = {
    name: rank
    for rank, name in enumerate(
        ("sel_0p5", "sel_1", "sel_5", "sel_10", "sel_30", "sel_60", "sel_80", "sel_95", "sel_99")
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("lng_half", type=Path)
    parser.add_argument("trie_half", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--lng-all-points", type=Path)
    parser.add_argument("--trie-all-points", type=Path)
    parser.add_argument("--lng-manifest", type=Path)
    parser.add_argument("--trie-manifest", type=Path)
    parser.add_argument("--lng-hierarchy-manifest", type=Path)
    parser.add_argument("--trie-hierarchy-manifest", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def row_qps(row: dict[str, str]) -> float | None:
    """Read either experiment_core or legacy selection-sweep QPS columns."""
    return finite_float(row.get("warm_median_qps", "") or row.get("qps_warm_median", ""))


def row_recall(row: dict[str, str]) -> float | None:
    return finite_float(row.get("recall_min", "") or row.get("recall", ""))


def normalize_crossings(
    source_rows: list[dict[str, str]],
    point_rows: list[dict[str, str]],
    method_to_code: dict[str, str] = METHOD_TO_CODE,
) -> dict[tuple[str, str], dict[str, object]]:
    """Normalize both summary schemas and synthesize explicit NC cells.

    The legacy summarizer emits only successful crossings. Its all-points file
    is therefore the authority that a method/workload was measured but did not
    cross the Recall threshold. The newer experiment_core schema emits those
    unavailable rows directly.
    """
    keyed: dict[tuple[str, str], dict[str, object]] = {}
    for row in source_rows:
        method = row.get("method", "")
        if method not in method_to_code:
            continue
        normalized: dict[str, object] = dict(row)
        qps = row_qps(row)
        normalized["warm_median_qps"] = "" if qps is None else qps
        normalized["recall_min"] = row.get("recall_min", "") or row.get("recall", "")
        normalized["status"] = row.get("status", "") or ("complete" if qps is not None else "unavailable")
        keyed[(row["workload"], method_to_code[method])] = normalized

    for point in point_rows:
        method = point.get("method", "")
        if method not in method_to_code:
            continue
        key = (point["workload"], method_to_code[method])
        if key in keyed:
            continue
        keyed[key] = {
            "workload": point["workload"],
            "method": method,
            "mean_selectivity": point.get("mean_selectivity", ""),
            "status": "unavailable",
            "warm_median_qps": "",
            "lsearch": "",
            "recall_min": "",
            "reason": "no measured Lsearch reaches Recall 0.9",
        }
    return keyed


def manifest_binary_hashes(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    return {
        str(run["search_binary_sha256"])
        for run in payload.get("runs", [])
        if run.get("status") == "complete" and run.get("search_binary_sha256")
    }


def manifest_build_hashes(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    return {
        str(run["source_provenance"]["build_binary_sha256"])
        for run in payload.get("runs", [])
        if run.get("status") == "complete"
        and run.get("source_provenance", {}).get("build_binary_sha256")
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def display_code(code: str) -> str:
    return f"0L[{code}]" if len(code) == 1 else f"{len(code) - 1}L[{code[0]}|{code[1:]}]"


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_fixed_configurations(
    long_rows: list[dict[str, object]],
    codes: list[str],
) -> list[dict[str, object]]:
    """Rank fixed configurations without rewarding missing Recall crossings.

    Cross-workload QPS is not averaged directly. Each crossing is normalized by
    the best measured configuration for that workload. Aggregate ratios are
    emitted only for configurations that cross the target on the entire grid.
    """
    best_qps: dict[str, float] = {}
    baseline_qps: dict[str, float] = {}
    for workload in WORKLOAD_ORDER:
        values = [
            finite_float(str(row["warm_median_qps"]))
            for row in long_rows
            if row["workload"] == workload and row["status"] == "complete"
        ]
        available = [value for value in values if value is not None]
        if available:
            best_qps[workload] = max(available)
        baseline = next(
            (row for row in long_rows if row["workload"] == workload and row["topology_code"] == "L"),
            None,
        )
        if baseline is not None:
            value = finite_float(str(baseline["warm_median_qps"]))
            if value is not None:
                baseline_qps[workload] = value

    summaries: list[dict[str, object]] = []
    total = len(WORKLOAD_ORDER)
    for code in codes:
        rows = [row for row in long_rows if row["topology_code"] == code]
        ratios: list[float] = []
        baseline_ratios: list[float] = []
        wins = 0
        for row in rows:
            workload = str(row["workload"])
            qps = finite_float(str(row["warm_median_qps"]))
            if row["status"] != "complete" or qps is None or workload not in best_qps:
                continue
            ratio = qps / best_qps[workload]
            ratios.append(ratio)
            wins += int(math.isclose(ratio, 1.0, rel_tol=1e-12, abs_tol=1e-12))
            if workload in baseline_qps:
                baseline_ratios.append(qps / baseline_qps[workload])
        full_coverage = len(ratios) == total and len(baseline_ratios) == total
        summaries.append(
            {
                "topology_code": code,
                "configuration": display_code(code),
                "recall_crossings": len(ratios),
                "total_workloads": total,
                "oracle_wins": wins,
                "full_grid_geomean_fraction_of_oracle": geometric_mean(ratios) if full_coverage else "",
                "full_grid_worst_fraction_of_oracle": min(ratios) if full_coverage else "",
                "full_grid_geomean_speedup_vs_L0_LNG": geometric_mean(baseline_ratios) if full_coverage else "",
            }
        )
    ranked = sorted(
        (row for row in summaries if row["full_grid_geomean_fraction_of_oracle"] != ""),
        key=lambda row: float(row["full_grid_geomean_fraction_of_oracle"]),
        reverse=True,
    )
    ranks = {str(row["topology_code"]): rank for rank, row in enumerate(ranked, start=1)}
    for row in summaries:
        row["full_grid_rank"] = ranks.get(str(row["topology_code"]), "")
    return summaries


def result_cell(row: dict[str, object]) -> str:
    qps = finite_float(str(row["warm_median_qps"]))
    if qps is not None:
        return f"{qps:.2f}"
    maximum = finite_float(str(row["max_recall"]))
    return "NC" if maximum is None else f"NC ({maximum:.3f})"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.lng_manifest is None) != (args.trie_manifest is None):
        raise SystemExit("--lng-manifest and --trie-manifest must be supplied together")
    if (args.lng_hierarchy_manifest is None) != (args.trie_hierarchy_manifest is None):
        raise SystemExit(
            "--lng-hierarchy-manifest and --trie-hierarchy-manifest must be supplied together"
        )
    binary_sha256 = None
    if args.lng_manifest is not None and args.trie_manifest is not None:
        lng_hashes = manifest_binary_hashes(args.lng_manifest)
        trie_hashes = manifest_binary_hashes(args.trie_manifest)
        if len(lng_hashes) != 1 or lng_hashes != trie_hashes:
            raise SystemExit(
                f"factorial executable mismatch: LNG={sorted(lng_hashes)}, Trie={sorted(trie_hashes)}"
            )
        binary_sha256 = next(iter(lng_hashes))
    build_binary_sha256 = None
    if args.lng_hierarchy_manifest is not None and args.trie_hierarchy_manifest is not None:
        lng_build_hashes = manifest_build_hashes(args.lng_hierarchy_manifest)
        trie_build_hashes = manifest_build_hashes(args.trie_hierarchy_manifest)
        if len(lng_build_hashes) != 1 or lng_build_hashes != trie_build_hashes:
            raise SystemExit(
                "factorial build executable mismatch: "
                f"LNG={sorted(lng_build_hashes)}, Trie={sorted(trie_build_hashes)}"
            )
        build_binary_sha256 = next(iter(lng_build_hashes))
    source_rows = read_rows(args.lng_half) + read_rows(args.trie_half)
    point_rows: list[dict[str, str]] = []
    for path in (args.lng_all_points, args.trie_all_points):
        if path is not None and path.is_file():
            point_rows.extend(read_rows(path))
    max_recall: dict[tuple[str, str], float] = {}
    for row in point_rows:
        method = row.get("method", "")
        if method not in METHOD_TO_CODE:
            continue
        key = (row["workload"], METHOD_TO_CODE[method])
        value = row_recall(row)
        if value is not None:
            max_recall[key] = max(value, max_recall.get(key, float("-inf")))
    keyed = normalize_crossings(source_rows, point_rows)
    expected = {(workload, code) for workload in WORKLOAD_ORDER for code in METHOD_TO_CODE.values()}
    missing = sorted(expected - set(keyed))
    pending = sorted(
        key for key, row in keyed.items()
        if key in expected and "No such file or directory" in row.get("reason", "")
    )
    incomplete = missing + pending
    if incomplete and not args.allow_partial:
        raise SystemExit(
            f"missing {len(incomplete)} factorial artifacts; first: {incomplete[:5]}")

    long_rows: list[dict[str, object]] = []
    for workload in WORKLOAD_ORDER:
        baseline = keyed.get((workload, "L"), {})
        baseline_qps = finite_float(baseline.get("warm_median_qps", ""))
        for code in sorted(METHOD_TO_CODE.values(), key=lambda value: (len(value), value)):
            row = keyed.get((workload, code), {})
            qps = finite_float(row.get("warm_median_qps", "")) if row.get("status") == "complete" else None
            long_rows.append(
                {
                    "workload": workload,
                    "mean_selectivity": row.get("mean_selectivity", ""),
                    "topology_code": code,
                    "overlay_count": len(code) - 1,
                    "base_topology": "lng" if code[0] == "L" else "trie",
                    "entry_strategy": "optimized_lng" if code[0] == "L" else "trie",
                    "status": row.get("status", "missing"),
                    "lsearch": row.get("lsearch", ""),
                    "recall_min": row.get("recall_min", ""),
                    "max_recall": max_recall.get((workload, code), ""),
                    "warm_median_qps": "" if qps is None else qps,
                    "speedup_vs_L": "" if qps is None or not baseline_qps else qps / baseline_qps,
                    "method": row.get("method", ""),
                    "reason": row.get("reason", ""),
                }
            )
    long_fields = list(long_rows[0])
    write_csv(args.output_dir / "factorial_equal_recall.csv", long_rows, long_fields)

    best_rows: list[dict[str, object]] = []
    for workload in WORKLOAD_ORDER:
        candidates = [row for row in long_rows if row["workload"] == workload and row["status"] == "complete"]
        best_all = max(candidates, key=lambda row: float(row["warm_median_qps"])) if candidates else None
        result: dict[str, object] = {
            "workload": workload,
            "mean_selectivity": best_all["mean_selectivity"] if best_all else "",
            "best_topology": display_code(str(best_all["topology_code"])) if best_all else "",
            "best_qps": best_all["warm_median_qps"] if best_all else "",
            "best_speedup_vs_L": best_all["speedup_vs_L"] if best_all else "",
        }
        for depth in range(3):
            choices = [row for row in candidates if row["overlay_count"] == depth]
            winner = max(choices, key=lambda row: float(row["warm_median_qps"])) if choices else None
            result[f"best_{depth}L_topology"] = display_code(str(winner["topology_code"])) if winner else ""
            result[f"best_{depth}L_qps"] = winner["warm_median_qps"] if winner else ""
        best_rows.append(result)
    write_csv(args.output_dir / "best_by_selectivity.csv", best_rows, list(best_rows[0]))

    by_key = {
        (str(row["workload"]), str(row["topology_code"])): row
        for row in long_rows
    }
    pair_rows: list[dict[str, object]] = []
    for workload in WORKLOAD_ORDER:
        for code in sorted(METHOD_TO_CODE.values(), key=lambda value: (len(value), value)):
            for level in range(1, len(code)):
                if code[level] != "L":
                    continue
                trie_code = code[:level] + "T" + code[level + 1 :]
                left = next((row for row in long_rows if row["workload"] == workload and row["topology_code"] == code), None)
                right = next((row for row in long_rows if row["workload"] == workload and row["topology_code"] == trie_code), None)
                if not left or not right:
                    continue
                lq = finite_float(str(left["warm_median_qps"])) if left["status"] == "complete" else None
                tq = finite_float(str(right["warm_median_qps"])) if right["status"] == "complete" else None
                pair_rows.append(
                    {
                        "workload": workload,
                        "mean_selectivity": left["mean_selectivity"] or right["mean_selectivity"],
                        "changed_level": level,
                        "lng_code": code,
                        "trie_code": trie_code,
                        "lng_configuration": display_code(code),
                        "trie_configuration": display_code(trie_code),
                        "lng_status": left["status"],
                        "trie_status": right["status"],
                        "lng_qps": "" if lq is None else lq,
                        "trie_qps": "" if tq is None else tq,
                        "trie_over_lng": "" if tq is None or not lq else tq / lq,
                    }
                )
    write_csv(args.output_dir / "upper_trie_pairwise.csv", pair_rows, list(pair_rows[0]))

    base_rows: list[dict[str, object]] = []
    for workload in WORKLOAD_ORDER:
        for lng_code in ("L", "LL", "LT", "LLL", "LLT", "LTL", "LTT"):
            trie_code = "T" + lng_code[1:]
            left = by_key.get((workload, lng_code))
            right = by_key.get((workload, trie_code))
            if not left or not right:
                continue
            lq = finite_float(str(left["warm_median_qps"])) if left["status"] == "complete" else None
            tq = finite_float(str(right["warm_median_qps"])) if right["status"] == "complete" else None
            base_rows.append(
                {
                    "workload": workload,
                    "mean_selectivity": left["mean_selectivity"] or right["mean_selectivity"],
                    "upper_plan": lng_code[1:] or "none",
                    "lng_base_code": lng_code,
                    "trie_base_code": trie_code,
                    "lng_configuration": display_code(lng_code),
                    "trie_configuration": display_code(trie_code),
                    "lng_status": left["status"],
                    "trie_status": right["status"],
                    "lng_qps": "" if lq is None else lq,
                    "trie_qps": "" if tq is None else tq,
                    "trie_over_lng_matched_provider": "" if tq is None or not lq else tq / lq,
                }
            )
    write_csv(args.output_dir / "base_matched_pairwise.csv", base_rows, list(base_rows[0]))

    trie_effect_rows: list[dict[str, object]] = []
    for workload in WORKLOAD_ORDER:
        upper = [row for row in pair_rows if row["workload"] == workload]
        upper_comparable = [row for row in upper if finite_float(str(row["trie_over_lng"])) is not None]
        upper_wins = [row for row in upper_comparable if float(row["trie_over_lng"]) > 1.0]
        base = [row for row in base_rows if row["workload"] == workload]
        base_comparable = [
            row for row in base
            if finite_float(str(row["trie_over_lng_matched_provider"])) is not None
        ]
        base_wins = [
            row for row in base_comparable
            if float(row["trie_over_lng_matched_provider"]) > 1.0
        ]
        best_upper = max(upper_comparable, key=lambda row: float(row["trie_over_lng"]), default=None)
        best_base = max(
            base_comparable,
            key=lambda row: float(row["trie_over_lng_matched_provider"]),
            default=None,
        )
        trie_effect_rows.append(
            {
                "workload": workload,
                "mean_selectivity": next(
                    (row["mean_selectivity"] for row in long_rows if row["workload"] == workload), ""
                ),
                "upper_trie_wins": len(upper_wins),
                "upper_comparable_pairs": len(upper_comparable),
                "best_upper_change": "" if best_upper is None else (
                    f"L{best_upper['changed_level']}:{best_upper['lng_code']}->{best_upper['trie_code']}"
                ),
                "best_upper_trie_over_lng": "" if best_upper is None else best_upper["trie_over_lng"],
                "matched_trie_base_wins": len(base_wins),
                "base_comparable_pairs": len(base_comparable),
                "best_upper_plan_for_trie_base": "" if best_base is None else best_base["upper_plan"],
                "best_trie_base_over_lng_base": "" if best_base is None else best_base["trie_over_lng_matched_provider"],
            }
        )
    write_csv(args.output_dir / "trie_effect_summary.csv", trie_effect_rows, list(trie_effect_rows[0]))

    codes = sorted(METHOD_TO_CODE.values(), key=lambda value: (len(value), value))
    fixed_rows = summarize_fixed_configurations(long_rows, codes)
    write_csv(
        args.output_dir / "global_configuration_summary.csv",
        fixed_rows,
        list(fixed_rows[0]),
    )

    lines = [
        "# Amazon L/T topology factorial",
        "",
        "All QPS values are selected at minimum-over-repeat Recall@10 >= 0.90. ",
        "L0=L uses optimized-LNG entry; L0=T uses Trie entry. Ratios across L0 are ",
        "therefore matched-provider system comparisons, not pure base-edge causal effects.",
        "",
        "| Selectivity | Overall best | QPS | vs L0-LNG | Best 0L | Best 1L | Best 2L |",
        "|---:|:---:|---:|---:|:---:|:---:|:---:|",
    ]
    for row in best_rows:
        sel = 100.0 * float(row["mean_selectivity"]) if row["mean_selectivity"] else None
        speedup = finite_float(str(row["best_speedup_vs_L"]))
        qps = finite_float(str(row["best_qps"]))
        lines.append(
            f"| {fmt(sel, 1)}% | {row['best_topology'] or '--'} | {fmt(qps)} | "
            f"{fmt(speedup)}x | {row['best_0L_topology'] or '--'} | "
            f"{row['best_1L_topology'] or '--'} | {row['best_2L_topology'] or '--'} |"
        )
    lines.extend(
        [
            "",
            "Topology strings list L0 first, followed by upper levels; L means LNG and T means Trie.",
            "The screen has one cold and two warm repeats. No unpaired ratio is labeled as a confidence interval.",
            "",
            "## Where Trie helps",
            "",
            "Upper replacements hold the base, all other levels, and the entry provider fixed. "
            "Base replacements use matched providers and are end-to-end system comparisons.",
            "",
            "| Selectivity | Upper Trie wins | Best upper replacement | Best T/L | Trie-base wins | Best upper plan | Best matched T/L |",
            "|---:|---:|:---:|---:|---:|:---:|---:|",
        ]
    )
    for row in trie_effect_rows:
        sel = 100.0 * float(row["mean_selectivity"]) if row["mean_selectivity"] else None
        upper_ratio = finite_float(str(row["best_upper_trie_over_lng"]))
        base_ratio = finite_float(str(row["best_trie_base_over_lng_base"]))
        lines.append(
            f"| {fmt(sel, 1)}% | {row['upper_trie_wins']}/{row['upper_comparable_pairs']} | "
            f"{row['best_upper_change'] or '--'} | {fmt(upper_ratio, 3)}x | "
            f"{row['matched_trie_base_wins']}/{row['base_comparable_pairs']} | "
            f"{row['best_upper_plan_for_trie_base'] or '--'} | {fmt(base_ratio, 3)}x |"
        )
    lines.append("")
    lines.extend(
        [
            "The matched-provider base ratios above must not be attributed to base topology alone; "
            "the available campaign does not contain the same-entry cross-base hierarchy control.",
            "",
            "## Fixed configuration over the full grid",
            "",
            "QPS is normalized by the best measured configuration at each workload before aggregation. "
            "Geometric means and worst ratios are reported only for configurations with 9/9 Recall crossings.",
            "",
            "| Rank | Configuration | Crossings | Oracle wins | Geomean/oracle | Worst/oracle | Geomean vs 0L[L] |",
            "|---:|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        fixed_rows,
        key=lambda value: (
            value["full_grid_rank"] == "",
            int(value["full_grid_rank"]) if value["full_grid_rank"] != "" else 999,
            -int(value["recall_crossings"]),
        ),
    ):
        geomean_oracle = finite_float(str(row["full_grid_geomean_fraction_of_oracle"]))
        worst_oracle = finite_float(str(row["full_grid_worst_fraction_of_oracle"]))
        geomean_baseline = finite_float(str(row["full_grid_geomean_speedup_vs_L0_LNG"]))
        lines.append(
            f"| {row['full_grid_rank'] or '--'} | {row['configuration']} | "
            f"{row['recall_crossings']}/{row['total_workloads']} | {row['oracle_wins']} | "
            f"{fmt(geomean_oracle, 3)} | {fmt(worst_oracle, 3)} | {fmt(geomean_baseline, 3)}x |"
        )
    lines.append("")
    (args.output_dir / "README.md").write_text("\n".join(lines))

    latex = [r"\newcommand{\baseTopologyFactorialTables}{%"]
    panels = (
        (codes[:6], "Zero- and one-overlay configurations", "tab:base-topology-factorial-a"),
        (codes[6:], "Two-overlay configurations", "tab:base-topology-factorial-b"),
    )
    for panel_codes, title, label in panels:
        latex.extend(
            [
                r"\begin{table*}[htbp]",
                r"\centering\scriptsize",
                r"\caption{" + title + r" in the complete Amazon base-by-overlay topology screen at Recall@10 $\geq0.90$. A code $m$L[$B|U$] lists base topology $B$ before the separator and upper topologies $U$ from fine to coarse. $L$ is LNG and $T$ is Trie. The base uses its matched entry provider: optimized-LNG for $B=L$ and Trie for $B=T$. NC$(r)$ gives maximum minimum-warm-repeat Recall when there is no crossing in the shared measured budget.}",
                r"\label{" + label + "}",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{r" + "r" * len(panel_codes) + "}",
                r"\toprule",
                "Sel. & " + " & ".join(display_code(code) for code in panel_codes) + r" \\",
                r"\midrule",
            ]
        )
        for workload in WORKLOAD_ORDER:
            row0 = next(row for row in long_rows if row["workload"] == workload)
            sel = 100.0 * float(row0["mean_selectivity"])
            latex.append(
                f"{sel:.3f}\\% & "
                + " & ".join(result_cell(by_key[(workload, code)]) for code in panel_codes)
                + r" \\"
            )
        latex.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    latex.append("}")
    latex.append(r"\newcommand{\baseTopologyFactorialFindings}{%")
    latex.extend(
        [
            r"\paragraph{Complete base-by-overlay factorial.}",
            r"The workload-wise oracle selects \texttt{2L[T|LT]} at 0.5\%, 1\%, 5\%, and 95\%; "
            r"\texttt{0L[L]} at 10\% and 30\%; \texttt{1L[T|L]} at 60\% and 80\%; and "
            r"\texttt{2L[T|TT]} at 99\%.  Thus no static topology dominates every workload. "
            r"Among configurations crossing the Recall target at all nine selectivities, "
            r"\texttt{2L[T|LT]} ranks first by geometric mean of QPS normalized to the "
            r"per-workload oracle (0.868), but its worst fraction of the oracle is 0.482, "
            r"at 30\%.  The matched-provider base-only comparison is equally non-monotone: "
            r"\texttt{0L[T]} is 6.03$\times$, 2.77$\times$, and 18.43$\times$ faster at "
            r"0.5\%, 1\%, and 5\%, respectively, but is 0.60$\times$ at 10\%, does not "
            r"cross at 30\% ($R_{\max}=0.8882$), and is 0.62$\times$--0.33$\times$ at "
            r"60\%--95\% (0.56$\times$ at 99\%).  These are end-to-end system ratios "
            r"because the two bases use different compatible entry providers.",
            "",
            r"The controlled upper-level comparisons reveal a more stable role for Trie. "
            r"With a Trie base and LNG at level 1, changing only level 2 from LNG to Trie "
            r"(\texttt{2L[T|LL]}$\rightarrow$\texttt{2L[T|LT]}) improves QPS in eight of "
            r"nine workloads, by up to 1.251$\times$; the remaining 80\% point is effectively "
            r"tied at 0.998$\times$.  In contrast, changing only the finer level 1 from LNG "
            r"to Trie with level 2 fixed to LNG wins in five of nine workloads but loses "
            r"40.7\%--52.9\% at 10\%, 60\%, 80\%, and 95\%.  The evidence therefore favors "
            r"Trie as a sparse coarse overlay more consistently than as the fine navigation "
            r"topology, while retaining an LNG fallback for the 10\%--30\% region.",
            "",
            r"\begin{table*}[htbp]",
            r"\centering\small",
            r"\caption{Best measured configuration at each Amazon selectivity.  Each depth "
            r"column is optimized only within that depth; the overall column ranges over all "
            r"14 configurations.  Values are QPS at Recall@10 $\geq0.90$.}",
            r"\label{tab:base-topology-winners}",
            r"\begin{tabular}{r rr rr rr rr}",
            r"\toprule",
            r"Sel. & \multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{Best 0L} & "
            "\\multicolumn{2}{c}{Best 1L} & \\multicolumn{2}{c}{Best 2L} \\\\",
            " & Config. & QPS & Config. & QPS & Config. & QPS & Config. & QPS \\\\",
            r"\midrule",
        ]
    )
    for row in best_rows:
        sel = 100.0 * float(row["mean_selectivity"])
        latex.append(
            f"{sel:.1f}\\% & {row['best_topology']} & {float(row['best_qps']):.2f} & "
            f"{row['best_0L_topology']} & {float(row['best_0L_qps']):.2f} & "
            f"{row['best_1L_topology']} & {float(row['best_1L_qps']):.2f} & "
            f"{row['best_2L_topology']} & {float(row['best_2L_qps']):.2f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", "}"])
    (args.output_dir / "generated_topology_factorial.tex").write_text("\n".join(latex) + "\n")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lng_half": str(args.lng_half),
                "trie_half": str(args.trie_half),
                "inputs": [
                    {"path": str(path), "sha256": sha256(path)}
                    for path in (
                        args.lng_half,
                        args.trie_half,
                        args.lng_all_points,
                        args.trie_all_points,
                        args.lng_manifest,
                        args.trie_manifest,
                        args.lng_hierarchy_manifest,
                        args.trie_hierarchy_manifest,
                    )
                    if path is not None
                ],
                "expected_rows": len(expected),
                "observed_rows": len(set(keyed) & expected) - len(pending),
                "missing_rows": missing,
                "pending_rows": pending,
                "inference_scope": "descriptive_screen",
                "search_binary_sha256": binary_sha256,
                "build_binary_sha256": build_binary_sha256,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
