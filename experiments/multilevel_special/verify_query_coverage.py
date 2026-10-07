#!/usr/bin/env python3
"""Read-only exact AND coverage audit; only writes a new output directory."""
import argparse
import csv
import datetime
import hashlib
import json
import resource
import shlex
import signal
import sys
import time
from collections import Counter
from pathlib import Path

WORKLOADS = {
    "sel_0p5": "query_minlen5_avgsel05pct",
    "sel_1": "query_minlen5_avgsel1pct",
    "sel_5": "query_nested_avgsel5pct",
    "sel_10": "query_minlen3_avgsel10pct",
    "sel_30": "query_nested_avgsel30pct",
    "sel_60": "query_nested_avgsel60pct",
    "sel_80": "query_minlen1_nested_avgsel80pct",
    "sel_95": "query_minlen1_nested_avgsel95pct",
    "sel_99": "query_label1_empty_avgsel99pct",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def labels(text):
    stripped = text.strip()
    if not stripped:
        return ()
    fields = stripped.split(",") if "," in stripped else stripped.split()
    parsed = tuple(int(value.strip()) for value in fields)
    if any(value < 0 for value in parsed):
        raise ValueError("negative label")
    return parsed


def write_csv(path, columns, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-base-rows", type=int, default=602453)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 3300:
        parser.error("timeout must be between 1 and 3300 seconds")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("audit deadline")))
    signal.alarm(args.timeout_seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    sources = {}
    errors = []
    summary = {"status": "running", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "timeout_seconds": args.timeout_seconds, "no_ann_or_vector_distance_evaluation": True}

    def capture(path, role):
        path = Path(path)
        key = str(path)
        if key not in sources:
            stat = path.stat()
            sources[key] = {"path": key, "sha256": sha256(path), "bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "roles": []}
        if role not in sources[key]["roles"]:
            sources[key]["roles"].append(role)
        return sources[key]["sha256"]

    try:
        capture(Path(__file__).resolve(), "audit_script")
        base = args.data_root / "Amazon_base_labels.txt"
        base_sha = capture(base, "base_labels")
        warm = args.repo / "docs/papers/multilevel_ung/generated_data/warm_repeats/warm_repeat_points.csv"
        capture(warm, "frozen_crossings")
        with warm.open(newline="") as stream:
            crossings = list(csv.DictReader(stream))
        if len(crossings) != 95:
            errors.append(f"expected 95 frozen crossings, found {len(crossings)}")
        manifests = {}
        provenance_rows = []
        paths_by_workload = {}
        for crossing in crossings:
            run_dir = Path(crossing["detail_path"]).parent
            command = run_dir / "command.txt"
            capture(command, "historical_command")
            argv = shlex.split(command.read_text())
            def option(flag):
                return argv[argv.index(flag) + 1]
            observed_paths = (option("--query_label_file"), option("--query_bin_file"), option("--gt_file"))
            workload = crossing["workload"]
            prior = paths_by_workload.setdefault(workload, observed_paths)
            if prior != observed_paths:
                errors.append(f"inconsistent historical input paths for {workload}")
            expected_label = str(args.data_root / WORKLOADS[workload] / "Amazon_query_labels.txt")
            if observed_paths[0] != expected_label:
                errors.append(f"unexpected query label path: {observed_paths[0]}")
            manifest = run_dir.parent.parent.parent / "manifest_performance.json"
            manifest_sha = capture(manifest, "historical_run_manifest")
            if manifest not in manifests:
                manifests[manifest] = json.loads(manifest.read_text())["runs"]
            records = [r for r in manifests[manifest] if r.get("run_dir") == str(run_dir)]
            if not records:
                errors.append(f"missing historical run record: {run_dir}")
            expected_hashes = sorted({r.get("provenance", {}).get("base_labels_sha256", "") for r in records})
            matched = bool(expected_hashes) and expected_hashes == [base_sha]
            if not matched:
                errors.append(f"base SHA provenance mismatch: {run_dir}: {expected_hashes}")
            provenance_rows.append({"workload": workload, "method": crossing["method"],
                                    "Lsearch": crossing["lsearch"], "run_dir": str(run_dir),
                                    "manifest": str(manifest), "manifest_sha256": manifest_sha,
                                    "matching_run_records": len(records),
                                    "recorded_base_labels_sha256": ";".join(expected_hashes),
                                    "observed_base_labels_sha256": base_sha,
                                    "base_sha_matches": int(matched), "query_label_file": observed_paths[0],
                                    "query_bin_file": observed_paths[1], "gt_file": observed_paths[2]})
        write_csv(args.output / "historical_provenance.csv", list(provenance_rows[0]), provenance_rows)

        query_rows = []
        input_details = {}
        generation_checks = []
        duplicate_query_rows = []
        for workload, dirname in WORKLOADS.items():
            folder = args.data_root / dirname
            label_path = folder / "Amazon_query_labels.txt"
            profile_paths = sorted(folder.glob("profiled_*.csv"))
            if len(profile_paths) != 1:
                raise ValueError(f"expected exactly one profile: {folder}")
            profile = profile_paths[0]
            files = {"query_labels_sha256": label_path, "profile_sha256": profile,
                     "query_bin_sha256": Path(paths_by_workload[workload][1]),
                     "groundtruth_sha256": Path(paths_by_workload[workload][2])}
            hashes = {key: capture(path, f"{workload}:{key}") for key, path in files.items()}
            generation = folder / "generation_manifest.json"
            if generation.exists():
                capture(generation, "query_generation_manifest")
                data = json.loads(generation.read_text())
                for key, actual in hashes.items():
                    if key in data:
                        match = data[key] == actual
                        generation_checks.append({"workload": workload, "field": key, "recorded": data[key],
                                                  "observed": actual, "matches": match,
                                                  "manifest": str(generation)})
                        if not match:
                            errors.append(f"generation manifest SHA mismatch: {workload}:{key}")
            loaded = [labels(line) for line in label_path.read_text().splitlines()]
            with profile.open(newline="") as stream:
                profiles = list(csv.DictReader(stream))
            if len(loaded) != 1000 or len(profiles) != len(loaded):
                raise ValueError(f"row count mismatch for {workload}: {len(loaded)}, {len(profiles)}")
            for query_id, (raw_labels, annotation) in enumerate(zip(loaded, profiles)):
                canonical = tuple(sorted(set(raw_labels)))
                profile_labels = tuple(sorted(set(labels(annotation["labels"]))))
                if canonical != profile_labels:
                    errors.append(f"profile/query label mismatch: {workload}:{query_id}")
                if len(raw_labels) != len(canonical):
                    duplicate_query_rows.append({"workload": workload, "QueryID": query_id,
                                                 "labels": list(raw_labels)})
                query_rows.append({"workload": workload, "QueryID": query_id,
                                   "QuerySize": len(raw_labels), "predicate": canonical,
                                   "profile_coverage_count": int(annotation["coverage_count"]),
                                   "profile_labels_match": canonical == profile_labels})
            input_details[workload] = {"query_label_file": str(label_path), "profile_file": str(profile),
                                       "hashes": hashes, "generation_manifest_present": generation.exists()}

        predicates = sorted({row["predicate"] for row in query_rows})
        queried_labels = sorted({label for predicate in predicates for label in predicate})
        bytes_per_bitmap = (args.expected_base_rows + 7) // 8
        bitmap_bytes = {label: bytearray(bytes_per_bitmap) for label in queried_labels}
        all_label_counts = Counter()
        duplicates = []
        duplicate_occurrences = 0
        empty_base_rows = 0
        base_rows = 0
        # Check a deterministic spread of predicates by direct Python set inclusion
        # during the same full input scan, independently of bitmap intersections.
        check_indices = {0, len(predicates) - 1}
        check_indices.update(i * (len(predicates) - 1) // 11 for i in range(12))
        direct_sets = {predicates[i]: frozenset(predicates[i]) for i in sorted(check_indices)}
        direct_counts = {predicate: 0 for predicate in direct_sets}
        scan_hash = hashlib.sha256()
        print(f"BEGIN base scan; rows_expected={args.expected_base_rows}, labels={len(queried_labels)}, bitmap_bytes={len(queried_labels)*bytes_per_bitmap}", flush=True)
        with base.open("rb") as stream:
            for row_id, raw in enumerate(stream):
                if row_id >= args.expected_base_rows:
                    raise ValueError("more base rows than expected")
                scan_hash.update(raw)
                raw_labels = labels(raw.decode("utf-8"))
                unique = set(raw_labels)
                if not unique:
                    empty_base_rows += 1
                excess = len(raw_labels) - len(unique)
                if excess:
                    duplicate_occurrences += excess
                    duplicates.append({"base_row_id": row_id, "labels": list(raw_labels),
                                       "duplicate_occurrences": excess})
                all_label_counts.update(unique)
                byte, bit = divmod(row_id, 8)
                mask = 1 << bit
                for label in unique:
                    bitmap = bitmap_bytes.get(label)
                    if bitmap is not None:
                        bitmap[byte] |= mask
                for predicate, subset in direct_sets.items():
                    if subset.issubset(unique):
                        direct_counts[predicate] += 1
                base_rows += 1
                if base_rows % 200000 == 0:
                    print(f"SCANNED {base_rows}, elapsed={time.monotonic()-start:.2f}s", flush=True)
        if base_rows != args.expected_base_rows:
            raise ValueError(f"expected {args.expected_base_rows} base rows, observed {base_rows}")
        if scan_hash.hexdigest() != base_sha:
            errors.append("base file SHA changed between initial hash and exact scan")
        bitmaps = {}
        for label in queried_labels:
            bitmaps[label] = int.from_bytes(bitmap_bytes.pop(label), "little")
            if bitmaps[label].bit_count() != all_label_counts[label]:
                errors.append(f"single-label bitmap count mismatch: {label}")
        eligible_cache = {}
        for predicate in predicates:
            if not predicate:
                eligible_cache[predicate] = base_rows
                continue
            ordered = sorted(predicate, key=lambda label: all_label_counts[label])
            intersection = bitmaps[ordered[0]]
            for label in ordered[1:]:
                intersection &= bitmaps[label]
                if not intersection:
                    break
            eligible_cache[predicate] = intersection.bit_count()
        direct_check_rows = []
        for predicate, direct_count in direct_counts.items():
            bitmap_count = eligible_cache[predicate]
            match = direct_count == bitmap_count
            direct_check_rows.append({"labels": ",".join(map(str, predicate)), "direct_count": direct_count,
                                      "bitmap_count": bitmap_count, "matches": int(match)})
            if not match:
                errors.append(f"independent direct predicate check mismatch: {predicate}")

        output_rows = []
        comparisons = []
        workload_summaries = []
        for row in query_rows:
            count = eligible_cache[row["predicate"]]
            output = {"workload": row["workload"], "QueryID": row["QueryID"],
                      "QuerySize": row["QuerySize"], "labels": ",".join(map(str, row["predicate"])),
                      "eligible_count": count, "true_selectivity": format(count / base_rows, ".17g")}
            output_rows.append(output)
            comparisons.append({**output, "profile_coverage_count": row["profile_coverage_count"],
                                "coverage_delta": count-row["profile_coverage_count"],
                                "profile_labels_match": int(row["profile_labels_match"])})
        mismatches = [row for row in comparisons if row["coverage_delta"] != 0 or not row["profile_labels_match"]]
        if mismatches:
            errors.append(f"{len(mismatches)} query rows disagree with their profiles")
        for workload in WORKLOADS:
            group = [row for row in output_rows if row["workload"] == workload]
            values = [row["eligible_count"] for row in group]
            workload_summaries.append({"workload": workload, "num_queries": len(group),
                "unique_predicates": len({row["labels"] for row in group}),
                "empty_queries": sum(row["QuerySize"] == 0 for row in group),
                "single_label_queries": sum(row["QuerySize"] == 1 for row in group),
                "multiple_label_queries": sum(row["QuerySize"] > 1 for row in group),
                "min_eligible_count": min(values), "max_eligible_count": max(values),
                "mean_true_selectivity": sum(values) / len(group) / base_rows,
                "zero_eligible_queries": sum(value == 0 for value in values),
                "fewer_than_10_eligible_queries": sum(value < 10 for value in values),
                "profile_mismatches": sum(row["workload"] == workload for row in mismatches)})
        write_csv(args.output / "query_coverage_exact.csv", list(output_rows[0]), output_rows)
        write_csv(args.output / "profile_comparison.csv", list(comparisons[0]), comparisons)
        write_csv(args.output / "profile_mismatches.csv", list(comparisons[0]), mismatches)
        write_csv(args.output / "workload_summary.csv", list(workload_summaries[0]), workload_summaries)
        write_csv(args.output / "independent_direct_checks.csv", list(direct_check_rows[0]), direct_check_rows)
        write_json(args.output / "duplicate_label_audit.json", {
            "base_rows_with_duplicates": len(duplicates), "base_duplicate_occurrences": duplicate_occurrences,
            "base_duplicate_rows": duplicates, "query_rows_with_duplicates": len(duplicate_query_rows),
            "query_duplicate_rows": duplicate_query_rows, "empty_base_rows": empty_base_rows})
        write_json(args.output / "generation_manifest_checks.json", generation_checks)
        changed_sources = []
        for path, source in sources.items():
            after = sha256(path)
            source["sha256_after"] = after
            source["unchanged_during_audit"] = after == source["sha256"]
            if not source["unchanged_during_audit"]:
                changed_sources.append(path)
        if changed_sources:
            errors.append(f"input sources changed during audit: {changed_sources}")
        write_json(args.output / "source_hashes.json", list(sources.values()))
        summary.update({"status": "pass" if not errors else "mismatch", "errors": errors,
            "base_labels_path": str(base), "base_labels_sha256": base_sha,
            "base_rows": base_rows, "base_unique_labels": len(all_label_counts),
            "base_rows_with_duplicate_labels": len(duplicates), "base_duplicate_occurrences": duplicate_occurrences,
            "empty_base_rows": empty_base_rows, "query_rows_with_duplicate_labels": len(duplicate_query_rows),
            "query_rows": len(query_rows), "unique_predicates": len(predicates),
            "cached_predicate_reuses": len(query_rows)-len(predicates), "queried_unique_labels": len(queried_labels),
            "bitmap_storage_bytes": len(queried_labels)*bytes_per_bitmap,
            "profile_mismatches": len(mismatches), "historical_crossings_checked": len(provenance_rows),
            "historical_base_sha_matches": sum(row["base_sha_matches"] for row in provenance_rows),
            "historical_manifest_paths": [str(path) for path in manifests],
            "generation_hash_checks": len(generation_checks),
            "generation_hash_matches": sum(row["matches"] for row in generation_checks),
            "direct_full_scan_checks": len(direct_check_rows),
            "sources_hashed": len(sources), "sources_changed": changed_sources,
            "workload_inputs": input_details, "workloads": workload_summaries,
            "historical_query_identity_limit": "Run commands establish file paths; generation hashes corroborate available manifests. Exact run-time query-content snapshots are not asserted for workloads lacking recorded hashes.",
            "algorithm": "One bit per base row in each queried label posting; exact AND plus popcount, cached by sorted unique predicate; empty predicate count is N."})
    except Exception as error:
        summary.update({"status": "failed", "errors": errors + [repr(error)]})
        write_json(args.output / "source_hashes_partial.json", list(sources.values()))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        summary["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        outputs = [path for path in args.output.iterdir() if path.is_file()]
        summary["outputs"] = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs}
        write_json(args.output / "audit_summary.json", summary)
        print(json.dumps({key: summary.get(key) for key in ["status", "errors", "base_rows", "query_rows", "unique_predicates", "profile_mismatches", "historical_base_sha_matches", "generation_hash_matches", "elapsed_seconds", "peak_rss_kib_linux"]}), flush=True)
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
