#!/usr/bin/env python3

import unittest
from pathlib import Path

import prepare_deadline_build_campaign as deadline


def case(name: str, profile: str, role: str, repeat: int) -> dict:
    return {
        "name": name,
        "benchmark_profile": profile,
        "timing_role": role,
        "repeat": repeat,
    }


class DeadlineBuildPreparationTest(unittest.TestCase):
    def test_absolute_path_does_not_resolve_platform_symlinks(self) -> None:
        path = Path("/home/example/run")
        self.assertEqual(deadline.absolute_no_resolve(path), path)

    def test_timing_selection_keeps_one_cold_and_two_measured(self) -> None:
        config = {"cases": [
            case("p_cold_r0", "p", "cold", 0),
            case("p_measured_r0", "p", "measured", 0),
            case("p_measured_r1", "p", "measured", 1),
            case("p_measured_r2", "p", "measured", 2),
            case("q_cold_r0", "q", "cold", 0),
        ]}
        selected = deadline.select_cases(
            config, ("p",), None, measured_repeats=2,
            resource_profile=False)
        self.assertEqual(
            [row["name"] for row in selected],
            ["p_cold_r0", "p_measured_r0", "p_measured_r1"],
        )

    def test_hierarchy_selection_is_limited_to_auto_drh(self) -> None:
        config = {"cases": [
            case("single_t1024_lng_cpu_cold_r0", "cpu", "cold", 0),
            case("auto_drh_v1_cpu_cold_r0", "cpu", "cold", 0),
            case("auto_drh_v1_cpu_measured_r0", "cpu", "measured", 0),
        ]}
        selected = deadline.select_cases(
            config, ("cpu",), "auto_drh_v1", measured_repeats=1,
            resource_profile=False)
        self.assertEqual(
            [row["name"] for row in selected],
            ["auto_drh_v1_cpu_cold_r0", "auto_drh_v1_cpu_measured_r0"],
        )

    def test_resource_selection_keeps_one_measured_case(self) -> None:
        config = {"cases": [
            case("p_cold_r0", "p", "cold", 0),
            case("p_measured_r0", "p", "measured", 0),
            case("p_measured_r1", "p", "measured", 1),
        ]}
        selected = deadline.select_cases(
            config, ("p",), None, measured_repeats=2,
            resource_profile=True)
        self.assertEqual([row["name"] for row in selected], ["p_measured_r0"])

    def test_prepare_rewrites_root_and_records_protocol(self) -> None:
        source = {
            "purpose": "source",
            "output_root": "/old",
            "cases": [
                case("p_cold_r0", "p", "cold", 0),
                case("p_measured_r0", "p", "measured", 0),
                case("p_measured_r1", "p", "measured", 1),
            ],
        }
        result = deadline.prepare_config(
            source, Path("/new"), ("p",), None, 2, False)
        self.assertEqual(result["output_root"], "/new")
        self.assertEqual(result["campaign_protocol"]["cold_repeats"], 1)
        self.assertEqual(result["campaign_protocol"]["measured_repeats"], 2)
        self.assertEqual(source["output_root"], "/old")


if __name__ == "__main__":
    unittest.main()
