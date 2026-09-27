#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import prepare_drh_v2_ablation as target


class PrepareDrhV2AblationTest(unittest.TestCase):
    def test_builds_same_binary_three_way_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.json"
            source_path.write_text("{}\n", encoding="utf-8")
            binary = root / "search"
            binary.write_bytes(b"same binary")
            source = {
                "schema_version": 2,
                "method_schema": "orthogonal_v2",
                "output_root": str(root / "old"),
                "measurement_pass": "performance",
                "protocol": {"cold_repeats": 1, "measured_repeats": 2},
                "workloads": [{"name": "w", "mean_selectivity": 0.1}],
                "methods": [
                    {
                        "name": "plain",
                        "base_topology": "lng",
                        "entry_strategy": "optimized_lng",
                        "hierarchy_layers": [],
                        "special_block_search": False,
                        "env": {},
                    },
                    {
                        "name": "drh",
                        "base_topology": "lng",
                        "entry_strategy": "optimized_lng",
                        "hierarchy_layers": [
                            {"min_points": 1024, "topology": "lng"},
                            {"min_points": 16384, "topology": "trie"},
                        ],
                        "special_block_search": True,
                        "block_index": str(root / "blocks"),
                        "env": {"UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION": "1"},
                    },
                ],
            }
            result = target.make_ablation_config(
                source, source_path, binary,
                hashlib.sha256(binary.read_bytes()).hexdigest(), "abc123",
                root / "new", "plain", "drh", 262144,
            )

            self.assertEqual([method["name"] for method in result["methods"]],
                             ["plain", "drh", "drh_next_scale_mass"])
            self.assertNotIn(
                "UNG_SPECIAL_UPPER_MIN_COVERED_POINTS",
                result["methods"][1]["env"],
            )
            self.assertEqual(
                result["methods"][2]["env"]["UNG_SPECIAL_UPPER_MIN_COVERED_POINTS"],
                "262144",
            )
            self.assertEqual(result["drh_v2_provenance"]["rho"], 16)
            self.assertEqual(result["search_app"], str(binary))

    def test_rejects_non_multilevel_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.json"
            source_path.write_text(json.dumps({}), encoding="utf-8")
            binary = root / "search"
            binary.write_bytes(b"binary")
            source = {
                "methods": [
                    {"name": "plain", "hierarchy_layers": []},
                    {"name": "drh", "hierarchy_layers": [{"min_points": 1}]},
                ],
            }
            with self.assertRaisesRegex(ValueError, "multilevel"):
                target.make_ablation_config(
                    source, source_path, binary,
                    hashlib.sha256(binary.read_bytes()).hexdigest(), "abc123",
                    root / "new", "plain", "drh", 16,
                )


if __name__ == "__main__":
    unittest.main()
