#!/usr/bin/env python3
"""Measure resident memory of existing indexes without rebuilding or querying.

Each dataset/algorithm runs in a fresh process. Probe programs reuse the native
search sources' loaders and build flags, but have a separate, load-only main.
RSS is read from /proc/self/status after malloc_trim(0); baseline RSS is
subtracted. Static filtering data used by the adapters is included; query data,
search scratch, build buffers and redundant source vectors are excluded.
"""

import argparse
import csv
import json
import os
import re
import shlex
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from summarize_index_build import ALGORITHMS, DATASETS, GIB, write_csv


COMMON = r'''
#include <malloc.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cstdlib>
static unsigned long long resident_bytes() {
    malloc_trim(0);
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input, line))
        if (line.rfind("VmRSS:", 0) == 0)
            return std::stoull(line.substr(6)) * 1024;
    throw std::runtime_error("VmRSS is unavailable");
}
static void finish_probe(unsigned long long baseline) {
    auto loaded = resident_bytes();
    std::cout << "MEMORY_PROBE baseline_bytes=" << baseline
              << " loaded_bytes=" << loaded
              << " delta_bytes=" << (loaded - baseline) << std::endl;
    std::cout.flush(); std::fflush(nullptr);
    std::_Exit(0); // Keep the loaded objects alive until after the sample.
}
'''

MAIN_HEADER = r'''
int main(int argc, char **argv) {
    if (argc != 7) return 2;
    const std::string folder = argv[1], labels = argv[2], algorithm = argv[5], dataset = argv[6];
    const auto dim = std::stoi(argv[3]);
    const auto count = std::stoull(argv[4]);
    const auto baseline = resident_bytes();
'''

CURATOR = r'''
import ctypes, gc, json, sys
from pathlib import Path
project = Path(sys.argv[1])
sys.path.insert(0, str(project / 'experiments/curator_baseline'))
from curator_backend import CuratorBackend
from curator_project_adapter import load_label_sets, build_label_inverted_index
folder, labels = Path(sys.argv[2]), Path(sys.argv[3])
info = json.loads((folder / 'curator_meta.json').read_text())
backend = CuratorBackend.from_config(info['dim'], dict(
    build=info['build'], curator_repo=str(project/'thirdparty/curator-v2')))
libc = ctypes.CDLL(None)
def rss():
    gc.collect()
    libc.malloc_trim(0)
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:'))) * 1024
baseline = rss()
backend.load(folder)
base_labels = load_label_sets(labels)
label_index = build_label_inverted_index(base_labels)
loaded = rss()
print(f'MEMORY_PROBE baseline_bytes={baseline} loaded_bytes={loaded} delta_bytes={loaded-baseline}', flush=True)
'''


def compile_probes(project, output, env, only_names=None):
    specs = {
        "UNG": (
            "build_ung_multilevel_trie_gpu/apps/CMakeFiles/search_UNG_index.dir",
            None,
            '#include "uni_nav_graph.h"\n#include "navix_index.h"\n',
            r'''if (algorithm == "NaviX") {
                ANNS::StandaloneNavixIndex index;
                index.load(folder + "/", "float");
                finish_probe(baseline);
            } else {
                ANNS::UniNavGraph index;
                index.load(folder + "/", "", "float", "", "", dataset);
                if (algorithm == "PTNC") index.load_special_block_index(folder + "/../block_index_files/");
                finish_probe(baseline);
            }'''),
        "ACORN": (
            "build_acorn_rel/demos/CMakeFiles/test_acorn.dir", "ACORN/demos/test_acorn.cpp", "",
            r'''auto metadata = load_ab_muti(dataset, 80, "rand", count, labels);
            auto *index = dynamic_cast<faiss::IndexACORNFlat *>(faiss::read_index((folder+"/acorn.index").c_str()));
            if (!index) throw std::runtime_error("invalid ACORN index");
            index->set_metadata(metadata);
            auto inverted = load_inverted_index(folder+"/acorn.index.inverted_index");
            index->set_inverted_index(inverted);
            finish_probe(baseline);'''),
        "Milvus": (
            "Knowhere/project_adapter/build/CMakeFiles/milvus_hnsw_search_sweep.dir",
            "Knowhere/project_adapter/search_sweep.cpp", "",
            r'''auto index = milvus_adapter::load_index(folder+"/index.bin");
            auto filters = milvus_adapter::load_filter_index(folder+"/filters.bin");
            finish_probe(baseline);'''),
        "RWalks": (
            "RWalks/project_adapter/build/CMakeFiles/rwalks_search_sweep.dir",
            "RWalks/project_adapter/search_sweep.cpp", "",
            r'''auto schema = rwalks_adapter::load_schema(folder+"/index.rwalks.labels.txt");
            hnswlib::L2Space space(dim, schema.size());
            auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, folder+"/index.bin", false, 0, false);
            finish_probe(baseline);'''),
        "PathSeer": (
            "PathSeer/project_adapter/build/CMakeFiles/pathseer_search_sweep.dir",
            "PathSeer/project_adapter/search_sweep.cpp", "",
            r'''std::unique_ptr<faiss::Index> index(faiss::read_index((folder+"/index.bin").c_str()));
            auto base_labels = project_adapter::load_label_rows(labels);
            finish_probe(baseline);'''),
        "FAVOR": (
            "FAVOR/build/app/CMakeFiles/search_sweep.dir", "FAVOR/app/search_sweep.cpp", "",
            r'''hnswlib::L2Space space(dim);
            BaseSet schema;
            schema.load_label_schema(folder+"/index.bin.labels");
            favor::FAVOR<float> index(&space, folder+"/index.bin", false, count, 20);
            finish_probe(baseline);'''),
    }
    binaries, commands = {}, []
    for name, (relative, original, includes, body) in specs.items():
        if only_names is not None and name not in only_names:
            continue
        target = project / relative
        flags = dict(re.findall(r"^(CXX_\w+) = (.*)$", (target / "flags.make").read_text(), re.M))
        source = output / (name + "_load_probe.cpp")
        original_include = (f'#define main original_search_main\n#include "{project/original}"\n#undef main\n'
                            if original else includes)
        source.write_text(original_include + COMMON + MAIN_HEADER + body + "\nreturn 0;\n}\n")
        object_path, binary = output / (name + ".o"), output / (name + "_load_probe")
        compile_command = ["/usr/bin/c++"]
        for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
            compile_command.extend(shlex.split(flags.get(key, "")))
        compile_command.extend(["-Wno-error", "-c", str(source), "-o", str(object_path)])
        link_command = shlex.split((target / "link.txt").read_text())
        for pos, word in enumerate(link_command):
            if word.endswith(".cpp.o"):
                link_command[pos] = str(object_path)
        link_command[link_command.index("-o") + 1] = str(binary)
        working = target.parent.parent
        commands.append(dict(algorithm=name, compile=compile_command, link=link_command, cwd=str(working)))
        with (output / (name + "_compile.log")).open("w") as log:
            subprocess.run(compile_command, cwd=working, env=env, stdout=log, stderr=log, check=True)
            subprocess.run(link_command, cwd=working, env=env, stdout=log, stderr=log, check=True)
        binaries[name] = binary
        print(f"Compiled {name} load-only probe", flush=True)
    command_file = output / "build_commands.json"
    if only_names is not None and command_file.exists():
        commands = [c for c in json.loads(command_file.read_text())
                    if c["algorithm"] not in only_names] + commands
    command_file.write_text(json.dumps(commands, indent=2) + "\n")
    return binaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[3]
    parser.add_argument("--output-dir", type=Path, default=workspace / "FilterVectorResult/index_build_summary")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS))
    parser.add_argument("--reuse-probes", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Remeasure the selected dataset/algorithm pairs")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent independent load processes; requires RAM for all active indexes")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    project = workspace / "FilterVectorCode_multilevel_special"
    output = args.output_dir.resolve()
    probes = output / "memory_probes"
    probes.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["LD_LIBRARY_PATH"] = ":".join([
        str(workspace / "OpenBLAS"), str(workspace / "openblas/lib"),
        str(workspace / "boost_1_89_0/lib"), str(workspace / "zlib/lib"),
        "/usr/local/cuda-12.4/lib64", env.get("LD_LIBRARY_PATH", "")])
    binaries = ({name: probes / (name + "_load_probe")
                 for name in ("UNG", "ACORN", "Milvus", "RWalks", "PathSeer", "FAVOR")}
                if args.reuse_probes else compile_probes(project, probes, env))
    missing = [name for name, path in binaries.items() if not path.exists()]
    if missing:
        binaries.update(compile_probes(project, probes, env, only_names=missing))
    curator = probes / "Curator_load_probe.py"
    curator.write_text(CURATOR)
    destination = output / "loaded_memory_details.csv"
    records = []
    if destination.exists():
        with destination.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    if args.refresh:
        records = [r for r in records if not
                   (r["dataset"] in args.datasets and r["algorithm"] in args.algorithms)]
    pending = []
    for dataset in args.datasets:
        data = workspace / "FilterVectorData" / dataset
        with (data / f"{dataset}_base.bin").open("rb") as handle:
            count, dim = struct.unpack("<II", handle.read(8))
        for algorithm in args.algorithms:
            if any(row["dataset"] == dataset and row["algorithm"] == algorithm for row in records):
                continue
            index = workspace / "FilterVectorResult" / dataset / "index" / ALGORITHMS[algorithm] / "index_files"
            labels = data / f"{dataset}_base_labels.txt"
            if algorithm == "Curator":
                command = [str(project / ".venv_curator/bin/python"), str(curator), str(project), str(index), str(labels)]
            else:
                key = "UNG" if algorithm in ("UNG", "PTNC", "NaviX") else algorithm
                if algorithm == "FAVOR":
                    with (index / "index.bin").open("rb") as handle:
                        if handle.read(8) != b"FAVROA01":
                            raise RuntimeError(f"FAVOR must use CRoaring; run ensure_favor_roaring_indexes.py first: {index}")
                command = [str(binaries[key]), str(index), str(labels), str(dim), str(count), algorithm, dataset]
            pending.append((dataset, algorithm, command))

    def measure(task):
        dataset, algorithm, command = task
        log_path = probes / f"{dataset}_{algorithm}_load.log"
        started = time.monotonic()
        print(f"Loading {dataset}/{algorithm} ...", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError(f"load probe exited {result.returncode}: {log_path}")
        match = re.search(r"MEMORY_PROBE baseline_bytes=(\d+) loaded_bytes=(\d+) delta_bytes=(\d+)", log_path.read_text())
        if not match:
            raise RuntimeError(f"No memory sample: {log_path}")
        baseline, loaded, delta = map(int, match.groups())
        assert loaded > baseline and delta == loaded - baseline
        return dict(dataset=dataset, algorithm=algorithm, baseline_rss_bytes=baseline,
                    loaded_rss_bytes=loaded, index_rss_delta_bytes=delta,
                    index_rss_delta_gib=delta/GIB, load_wall_seconds=time.monotonic()-started,
                    measured_at_utc=datetime.now(timezone.utc).isoformat(), source=str(log_path))

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(measure, task) for task in pending]):
            row = future.result()
            records.append(row)
            write_csv(destination, records)
            print(f"Loaded {row['dataset']}/{row['algorithm']}: {row['index_rss_delta_gib']:.4f} GiB", flush=True)
    records.sort(key=lambda r: (DATASETS.index(r["dataset"]), list(ALGORITHMS).index(r["algorithm"])))
    write_csv(destination, records)
    matrix = [dict(dataset=dataset, **{algorithm: next((
        round(float(r["index_rss_delta_gib"]), 6) for r in records
        if r["dataset"] == dataset and r["algorithm"] == algorithm), "")
        for algorithm in ALGORITHMS}) for dataset in DATASETS]
    write_csv(output / "loaded_memory_gib.csv", matrix)
    build_details = output / "index_build_details.csv"
    if build_details.exists():
        with build_details.open(encoding="utf-8-sig", newline="") as handle:
            build_rows = {(r["dataset"], r["algorithm"]): r for r in csv.DictReader(handle)}
        combined = []
        for row in records:
            build = build_rows[(row["dataset"], row["algorithm"])]
            combined.append(dict(dataset=row["dataset"], algorithm=row["algorithm"],
                                 index_format="CRoaring (FAVROA01)" if row["algorithm"] == "FAVOR" else "",
                                 loaded_memory_gib=row["index_rss_delta_gib"],
                                 loaded_memory_bytes=row["index_rss_delta_bytes"],
                                 build_time_seconds=build["build_time_seconds"],
                                 memory_source=row["source"], build_time_source=build["build_time_source"]))
        summary_path = output / "memory_and_build_summary.csv"
        if summary_path.exists():
            # This summary may contain manually adjusted build times and newer
            # sizeof/capacity accounting. Refresh RSS cells only; retain both.
            with summary_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, summary_rows = reader.fieldnames, list(reader)
            by_key = {(r["dataset"], r["algorithm"]): r for r in combined}
            for old in summary_rows:
                new = by_key.get((old["dataset"], old["algorithm"]))
                if new is None or old.get("memory_method", "").startswith("search_sizeof_capacity") or \
                        "/search_memory_probes/" in old.get("memory_source", ""):
                    continue
                if old.get("memory_method") == "rss_delta_minus_base_vectors":
                    # Preserve the user's vector-free reporting scope.
                    structure_bytes = int(new["loaded_memory_bytes"]) - int(old["excluded_vector_bytes"])
                    old["loaded_memory_bytes"] = str(structure_bytes)
                    old["loaded_memory_gib"] = str(structure_bytes / GIB)
                    continue
                if old.get("memory_method") == "rss_delta_including_vectors":
                    old["index_structure_bytes"] = str(int(new["loaded_memory_bytes"]) - int(old["included_vector_bytes"]))
                for key in ("loaded_memory_gib", "loaded_memory_bytes", "memory_source"):
                    old[key] = new[key]
            write_csv(summary_path, summary_rows, fields)
            if any(r.get("memory_scope") == "index_structures_and_stored_vectors" for r in summary_rows):
                from exclude_vectors_from_memory_summary import update_summary
                excluded = {r['algorithm'] for r in summary_rows
                            if r.get('memory_scope') == 'index_structures_only'}
                update_summary(include_vectors=True, excluded_vector_algorithms=excluded)
        else:
            write_csv(summary_path, combined)
        favor_rows = [row for row in combined if row["algorithm"] == "FAVOR"]
        if favor_rows:
            write_csv(output / "favor_roaring_memory_build.csv", favor_rows)
    (output / "loaded_memory_methodology.json").write_text(json.dumps(dict(
        metric="index_rss_delta_bytes = VmRSS after load - VmRSS before load; fresh process for each pair",
        units="GiB = 2^30 bytes", completed_records=len(records),
        max_concurrent_load_processes=args.jobs,
        scope="Vectors, index structures and persistent filter support used by current search adapters; no queries, query scratch, result arrays or redundant source vector copies.",
        filters={"Milvus":"filters.bin", "RWalks":"schema and index_attr.bin (native loader)",
                 "FAVOR":"CRoaring (FAVROA01) native persisted attribute representation plus BaseSet label-schema map; legacy bitset indexes are rejected", "PathSeer":"base_labels rows",
                 "ACORN":"base metadata and inverted index (the setter copies inverted postings, whose local original is also retained by original search main)",
                 "Curator":"base frozensets and indexed-mode label postings (default search filter_mode)",
                 "UNG":"native load", "NaviX":"native load", "PTNC":"native base load plus independent block load"},
        procedure="Existing native source/loaders and build flags, separate probe main. One loading thread. malloc_trim(0) before both RSS samples; Python additionally runs gc.collect(). Process exits after sample without querying. Dynamic libraries/interpreter present before load are subtracted.",
        caveats="Resident memory, not allocated capacity or peak RSS. Lazy untouched pages are not included. Includes retained allocator/runtime overhead after trimming; small runtime/library initialization differences remain. Existing construction times are separate historical measurements."
    ), ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(records)} measured memory records", flush=True)


if __name__ == "__main__":
    main()
