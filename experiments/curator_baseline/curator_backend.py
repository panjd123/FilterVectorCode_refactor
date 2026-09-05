#!/usr/bin/env python3

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


def import_curator_class(curator_repo: str | None) -> type:
    if curator_repo:
        repo_path = str(Path(curator_repo).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        module = importlib.import_module("indexes.curator")
        return module.Curator
    except Exception as exc:
        raise ImportError(
            "Unable to import Curator-v2. Install its FAISS Python extension and set "
            "`curator_repo` in the config to the hatsu3/curator-v2 checkout."
        ) from exc


class CuratorBackend:
    def __init__(self, curator_cls: type, params: dict[str, Any]):
        self.params = dict(params)
        init_params = dict(params)
        nlist = int(init_params.pop("nlist"))
        init_params.pop("num_threads", None)
        dim = int(init_params.pop("d", init_params.pop("dim", 0)))
        if dim <= 0:
            dim = int(params.get("dimension", 0))
        if dim <= 0:
            raise ValueError("CuratorBackend requires positive dimension `d`")
        self.index = curator_cls(dim, nlist, **init_params)

    @classmethod
    def from_config(cls, dim: int, cfg: dict) -> "CuratorBackend":
        params = dict(cfg.get("build", {}))
        params.update(cfg.get("curator", {}))
        params["d"] = dim
        params.setdefault("nlist", 1024)
        curator_cls = import_curator_class(cfg.get("curator_repo"))
        return cls(curator_cls, params)

    def build(self, base_vectors: np.ndarray, base_ids: np.ndarray) -> None:
        vectors = np.asarray(base_vectors, dtype=np.float32)
        ids = np.asarray(base_ids, dtype=np.uint32)
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(
                f"base vector/id count mismatch: {vectors.shape[0]} vectors, {ids.shape[0]} ids"
            )
        self.index.train(vectors)
        for row, vector_id in enumerate(ids):
            self.index.create(vectors[row], int(vector_id))

    @staticmethod
    def artifact_path(index_dir: Path | str) -> Path:
        return Path(index_dir) / "curator.index"

    def save(self, index_dir: Path | str) -> None:
        index_path = self.artifact_path(index_dir)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(self.index, "index") or not hasattr(self.index.index, "save"):
            raise RuntimeError(
                "Curator-v2 FAISS extension does not expose MultiTenantIndexIVFHierarchical.save(). "
                "Rebuild the bundled FAISS extension with the project setup script."
            )
        try:
            self.index.index.save(str(index_path))
        except Exception:
            if index_path.exists():
                index_path.unlink()
            raise

    def load(self, index_dir: Path | str) -> None:
        index_path = self.artifact_path(index_dir)
        if not hasattr(self.index, "index") or not hasattr(self.index.index, "load"):
            raise RuntimeError(
                "Curator-v2 FAISS extension does not expose MultiTenantIndexIVFHierarchical.load(). "
                "Rebuild the bundled FAISS extension with the project setup script."
            )
        self.index.index.load(str(index_path))

    def memory_usage_bytes(self) -> int:
        if hasattr(self.index, "index") and hasattr(self.index.index, "memory_usage_estimate"):
            return int(self.index.index.memory_usage_estimate())
        return 0

    def set_search_budget(self, budget: int, parameter: str = "search_ef") -> None:
        parameter = str(parameter or "search_ef")
        if hasattr(self.index, "search_params"):
            params = dict(getattr(self.index, "search_params"))
            params[parameter] = int(budget)
            setattr(self.index, "search_params", params)
        elif hasattr(self.index, parameter):
            setattr(self.index, parameter, int(budget))
        else:
            raise RuntimeError(f"Curator-v2 backend does not expose search parameter `{parameter}`")

    def enable_stats_tracking(self, enable: bool = True) -> None:
        if hasattr(self.index, "enable_stats_tracking"):
            self.index.enable_stats_tracking(enable)
            return
        inner_index = getattr(self.index, "index", None)
        if inner_index is not None and hasattr(inner_index, "set_profiling_enabled"):
            inner_index.set_profiling_enabled(enable)
            return
        raise RuntimeError(
            "Curator-v2 backend does not expose search profiling; rebuild the bundled FAISS extension"
        )


    def prepare_filter(self, qualified_ids: np.ndarray) -> dict[str, Any]:
        qualified = np.asarray(qualified_ids, dtype=np.uint32)
        prepared: dict[str, Any] = {
            "qualified_ids": qualified,
            "qualified_labels_count": int(qualified.size),
        }
        inner_index = getattr(self.index, "index", None)
        if qualified.size > 0 and inner_index is not None and hasattr(inner_index, "get_label_to_vid_mapping"):
            vids = np.asarray(inner_index.get_label_to_vid_mapping(qualified), dtype=np.uint64)
            valid_vids = vids[vids != np.iinfo(np.uint64).max]
            prepared["sorted_vids"] = np.sort(valid_vids)
        return prepared

    def search(
        self, query: np.ndarray, k: int, qualified_ids: np.ndarray | dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        sorted_vids = None
        if isinstance(qualified_ids, dict):
            qualified = np.asarray(qualified_ids.get("qualified_ids", []), dtype=np.uint32)
            if "sorted_vids" in qualified_ids:
                sorted_vids = np.asarray(qualified_ids["sorted_vids"], dtype=np.uint64)
            qualified_count = int(qualified_ids.get("qualified_labels_count", qualified.size))
        else:
            qualified = np.asarray(qualified_ids, dtype=np.uint32)
            qualified_count = int(qualified.size)

        if qualified_count == 0 or (sorted_vids is not None and sorted_vids.size == 0):
            return np.full(k, -1, dtype=np.int64), {
                "qualified_labels_count": 0,
                "visited_points": 0,
                "visited_edges": 0,
                "distance_computations": 0,
            }

        query_batch = np.asarray(query, dtype=np.float32)[None]
        inner_index = getattr(self.index, "index", None)
        if sorted_vids is not None and inner_index is not None and hasattr(inner_index, "search_with_bitmap_filter_optimized"):
            raw = inner_index.search_with_bitmap_filter_optimized(query_batch, k, sorted_vids)
        elif hasattr(self.index, "search_with_bitmap_filter"):
            raw = self.index.search_with_bitmap_filter(query_batch[0], k, qualified)
        elif hasattr(self.index, "search_with_bitmap_filter_optimized"):
            raw = self.index.search_with_bitmap_filter_optimized(query_batch, k, qualified)
        else:
            raise RuntimeError("Curator-v2 backend does not expose bitmap-filter search")

        ids = raw[1] if isinstance(raw, tuple) and len(raw) == 2 else raw
        out = np.asarray(ids[0], dtype=np.int64) if np.asarray(ids).ndim == 2 else np.asarray(ids, dtype=np.int64)
        if out.shape[0] < k:
            padded = np.full(k, -1, dtype=np.int64)
            padded[: out.shape[0]] = out
            out = padded
        elif out.shape[0] > k:
            out = out[:k]

        profile: dict[str, Any] = {}
        if hasattr(self.index, "get_last_search_profile"):
            profile.update(self.index.get_last_search_profile())
        profile["qualified_labels_count"] = qualified_count
        for key in ("visited_points", "visited_edges", "distance_computations"):
            if key not in profile:
                raise RuntimeError(
                    f"Curator-v2 search profile is missing `{key}`; rebuild the bundled FAISS extension"
                )
        return out, profile
