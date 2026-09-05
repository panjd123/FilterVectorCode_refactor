# Curator-v2 Python/FAISS Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Curator-v2 Python/FAISS baseline runner that can evaluate current project datasets, query tasks, ground truth, and result CSVs.

**Architecture:** Keep Curator outside the UNG C++ binaries. Add a project-format adapter for vectors, labels, filters, recall, and CSV output; add a Curator backend wrapper that imports `indexes.curator.Curator` from a configured Curator-v2 checkout; add a config-driven runner under `experiments/curator_baseline/`.

**Tech Stack:** Python 3, NumPy, optional Curator-v2 FAISS Python package, unittest/pytest-compatible tests, existing project `.bin`/label/GT formats.

## Global Constraints

- Do not vendor Curator-v2 or FAISS into this repository.
- Preserve containment semantics: a base vector qualifies iff its base label set is a superset of the query label set.
- Keep result layout and summary columns compatible with existing experiment runners.
- Fail fast with an actionable message when Curator-v2 or its FAISS extension is unavailable.
- Do not modify existing UNG C++ search/build routes for this Python baseline.

---

### Task 1: Project Data Adapter

**Files:**
- Create: `experiments/curator_baseline/curator_project_adapter.py`
- Create: `experiments/curator_baseline/test_curator_project_adapter.py`

**Interfaces:**
- Produces: `load_project_bin(path: Path) -> np.ndarray`
- Produces: `load_label_sets(path: Path) -> list[frozenset[int]]`
- Produces: `qualified_ids_for_containment(base_labels: list[frozenset[int]], query_labels: frozenset[int]) -> np.ndarray`
- Produces: `load_groundtruth(path: Path, k: int) -> np.ndarray`
- Produces: `recall_at_k(results: np.ndarray, groundtruth: np.ndarray, k: int) -> float`

- [ ] **Step 1: Write failing adapter tests**

Create tests that write tiny project-format binary vectors, label files, and GT files. Assert loaded vector shape/dtype, empty label parsing, containment-qualified ids, and recall behavior with padded `-1`.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `pytest experiments/curator_baseline/test_curator_project_adapter.py -q`
Expected: FAIL because `curator_project_adapter` does not exist.

- [ ] **Step 3: Implement adapter functions**

Implement only the five public functions listed above. Use little-endian `<II` vector headers and `float32` vector payloads. For ground truth, support both project rows of repeated `<uint32,float32>` pairs and id-only fallback if needed.

- [ ] **Step 4: Run adapter tests and verify GREEN**

Run: `pytest experiments/curator_baseline/test_curator_project_adapter.py -q`
Expected: PASS.

### Task 2: Curator Backend Wrapper

**Files:**
- Create: `experiments/curator_baseline/curator_backend.py`
- Create: `experiments/curator_baseline/test_curator_backend.py`

**Interfaces:**
- Consumes: `load_project_bin`, `load_label_sets`, `qualified_ids_for_containment`
- Produces: `class CuratorBackend`
- Produces: `CuratorBackend.from_config(dim: int, cfg: dict) -> CuratorBackend`
- Produces: `CuratorBackend.build(base_vectors: np.ndarray, base_ids: np.ndarray) -> None`
- Produces: `CuratorBackend.search(query: np.ndarray, k: int, qualified_ids: np.ndarray) -> tuple[np.ndarray, dict]`
- Produces: `import_curator_class(curator_repo: str | None) -> type`

- [ ] **Step 1: Write failing backend tests**

Create a fake Curator class with `train`, `create`, `search_with_bitmap_filter`, and profiling methods. Assert the backend builds with project vector ids and searches with a `uint32` qualified-id array. Assert missing imports raise a message mentioning `curator_repo`.

- [ ] **Step 2: Run backend tests and verify RED**

Run: `pytest experiments/curator_baseline/test_curator_backend.py -q`
Expected: FAIL because `curator_backend` does not exist.

- [ ] **Step 3: Implement backend wrapper**

Implement dynamic import by prepending `curator_repo` to `sys.path` when provided, then importing `indexes.curator.Curator`. Use `search_with_bitmap_filter` first. Return padded ids and profile fields where available.

- [ ] **Step 4: Run backend tests and verify GREEN**

Run: `pytest experiments/curator_baseline/test_curator_backend.py -q`
Expected: PASS.

### Task 3: Curator Runner and Config

**Files:**
- Create: `experiments/curator_baseline/run_curator_baseline.py`
- Create: `experiments/curator_baseline/config.json`
- Create: `experiments/curator_baseline/test_run_curator_baseline.py`

**Interfaces:**
- Consumes: `CuratorBackend`, adapter functions
- Produces: `merged_build_cfg(cfg: dict, dataset_cfg: dict) -> dict`
- Produces: `merged_search_cfg(cfg: dict, dataset_cfg: dict) -> dict`
- Produces: `nprobe_values(search_cfg: dict) -> list[int]`
- Produces: `dataset_paths(cfg: dict, dataset_cfg: dict, search_cfg: dict) -> dict[str, Path | str]`
- Produces: `run_dataset(cfg: dict, dataset_cfg: dict) -> None`
- Produces: `main() -> int`

- [ ] **Step 1: Write failing runner tests**

Create a tiny dataset and monkeypatch `CuratorBackend.from_config` with a fake backend. Assert the runner writes `search_time_summary.csv`, `search_time_summary_qps.csv`, per-budget result files, and uses the same result directory naming pattern as NaviX.

- [ ] **Step 2: Run runner tests and verify RED**

Run: `pytest experiments/curator_baseline/test_run_curator_baseline.py -q`
Expected: FAIL because `run_curator_baseline` does not exist.

- [ ] **Step 3: Implement runner**

Implement config loading, path resolution, optional query `.fvecs` conversion and GT generation hooks, build/load metadata checks, query loop, recall calculation, timing, CSV writing, and QPS summary writing.

- [ ] **Step 4: Run runner tests and verify GREEN**

Run: `pytest experiments/curator_baseline/test_run_curator_baseline.py -q`
Expected: PASS.

### Task 4: Focused Verification

**Files:**
- All files in `experiments/curator_baseline/`

**Interfaces:**
- Consumes: all prior tasks
- Produces: verified local status for Curator baseline

- [ ] **Step 1: Run all Curator baseline tests**

Run: `pytest experiments/curator_baseline -q`
Expected: PASS.

- [ ] **Step 2: Run adjacent NaviX baseline tests**

Run: `pytest experiments/navix_baseline/test_run_navix_baseline.py -q`
Expected: PASS or report unrelated pre-existing failures.

- [ ] **Step 3: Inspect diff**

Run: `git status --short experiments/curator_baseline docs/superpowers/plans/2026-07-30-curator-v2-python-faiss-baseline.md`
Expected: only Curator baseline files and the plan are listed.
