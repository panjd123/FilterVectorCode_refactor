#!/usr/bin/env python3

import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_qps_recall_from_summary as plotter


class PlotQpsRecallSvgTest(unittest.TestCase):
    def test_svg_title_uses_dataset_name(self):
        series = [
            (
                "UNG",
                "Original UNG",
                "#4c78a8",
                "circle",
                [
                    {
                        "Lsearch": 1000,
                        "Average_Efs": 10.0,
                        "Average_Time_ms": 2.0,
                        "Average_Recall": 0.9,
                        "QPS": 500.0,
                    }
                ],
            )
        ]

        svg = plotter.make_svg(
            series,
            query_task="query_minlen1_cov10k",
            result_task="query_minlen1_cov10k",
            dataset="Genome",
            num_queries=1,
            sort_mode="time-column-only",
            logy=False,
            legend_outside=True,
        )

        self.assertIn("Genome query_minlen1_cov10k: QPS vs Recall", svg)
        self.assertNotIn("Amazon query_minlen1_cov10k: QPS vs Recall", svg)


class PlotQpsRecallMethodSpecTest(unittest.TestCase):
    def test_method_spec_can_auto_compose_result_task_from_query_task_and_lsearch(self):
        method = plotter.parse_method_spec(
            "FAVOR:FAVOR:#b279a2:diamond:query_selected_recall_advantage:10000:10000:200000"
        )

        self.assertEqual(method.method_dir, "FAVOR")
        self.assertEqual(method.label, "FAVOR")
        self.assertEqual(method.result_task("fallback_task"), "query_selected_recall_advantage_10000_10000_200000")

    def test_method_spec_can_override_result_task(self):
        method = plotter.parse_method_spec(
            "Curator:Curator:#9d755d:star:query_selected_recall_advantage:query_selected_recall_advantage_search_ef64_search_ef8192"
        )

        self.assertEqual(method.method_dir, "Curator")
        self.assertEqual(method.label, "Curator")
        self.assertEqual(method.result_task("fallback_task"), "query_selected_recall_advantage_search_ef64_search_ef8192")


class PlotQpsRecallReadSummaryTest(unittest.TestCase):
    def test_read_summary_accepts_curator_recall_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "search_time_summary.csv"
            summary_path.write_text(
                "Lsearch,Recall,Average_Time_ms\n"
                "64,0.669457,3770.205343\n"
            )

            rows = plotter.read_summary(summary_path)

        self.assertEqual(
            rows,
            [
                {
                    "Lsearch": 64,
                    "Average_Efs": 64.0,
                    "Average_Time_ms": 3770.205343,
                    "Average_Recall": 0.669457,
                }
            ],
        )

    def test_skip_first_lsearch_row_drops_smallest_lsearch(self):
        rows = [
            {"Lsearch": 1000, "Average_Recall": 0.3},
            {"Lsearch": 2000, "Average_Recall": 0.4},
            {"Lsearch": 3000, "Average_Recall": 0.5},
        ]
        self.assertEqual(plotter.skip_first_lsearch_row(rows), rows[1:])


class PlotQpsRecallConfigShellTest(unittest.TestCase):
    def test_config_shell_loops_over_query_tasks_and_expands_result_task(self):
        script_dir = Path(__file__).resolve().parent
        config_shell = script_dir / "plot_qps_recall_config.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_path = tmp_path / "argv.jsonl"
            recorder = tmp_path / "record_argv.py"
            recorder.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"with open({str(argv_path)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )

            config = {
                "script": str(recorder),
                "dataset": "Genome",
                "query_tasks": ["task_a", "task_b"],
                "paths": {
                    "results_root": str(tmp_path / "results"),
                    "data_root": str(tmp_path / "data"),
                    "output_dir": str(tmp_path / "plot" / "<result_task>"),
                },
                "methods": [
                    {
                        "method_dir": "Curator",
                        "label": "Curator",
                        "color": "#9d755d",
                        "marker": "triangle",
                        "result_task": "{query_task}_search_ef64_search_ef10240",
                    }
                ],
                "draw_options": {
                    "sort_mode": "time-column-only",
                    "legend_outside": True,
                    "skip_missing": False,
                    "write_log_scale_svg": True,
                    "threshold_step": 0.05,
                },
                "lsearch": {"start": 1000, "step": 1000, "end": 20000},
            }
            config_path = tmp_path / "plot_config.json"
            config_path.write_text(json.dumps(config))

            subprocess.run(["bash", str(config_shell), str(config_path)], check=True)

            invocations = [json.loads(line) for line in argv_path.read_text().splitlines()]
            self.assertEqual(len(invocations), 2)
            for argv, task in zip(invocations, ("task_a", "task_b")):
                self.assertIn(["--query-task", task], [argv[i:i + 2] for i in range(len(argv) - 1)])
                self.assertIn(
                    f"Curator:Curator:#9d755d:triangle:{task}:{task}_search_ef64_search_ef10240",
                    argv,
                )
                self.assertIn(str(tmp_path / "plot" / f"{task}_1000_1000_20000"), argv)

    def test_config_shell_passes_methods_from_json(self):
        script_dir = Path(__file__).resolve().parent
        config_shell = script_dir / "plot_qps_recall_config.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_path = tmp_path / "argv.json"
            recorder = tmp_path / "record_argv.py"
            recorder.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"open({str(argv_path)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            )

            config = {
                "script": str(recorder),
                "dataset": "Amazon",
                "query_task": "query_selected_time_advantage",
                "result_task": "query_selected_time_advantage_1000_1000_20000",
                "paths": {
                    "results_root": str(tmp_path / "results"),
                    "data_root": str(tmp_path / "data"),
                    "output_dir": str(tmp_path / "plot"),
                },
                "methods": [
                    {
                        "method_dir": "UNG__hybrid",
                        "label": "Original UNG",
                        "color": "#4c78a8",
                        "marker": "circle",
                        "skip_first_lsearch": True,
                    }
                ],
                "draw_options": {
                    "sort_mode": "time-column-only",
                    "legend_outside": True,
                    "skip_missing": False,
                    "write_log_scale_svg": True,
                    "threshold_step": 0.05,
                },
            }
            config_path = tmp_path / "plot_config.json"
            config_path.write_text(json.dumps(config))

            subprocess.run(["bash", str(config_shell), str(config_path)], check=True)

            argv = json.loads(argv_path.read_text())
            self.assertIn("--method", argv)
            self.assertIn("UNG__hybrid:Original UNG:#4c78a8:circle", argv)
            self.assertIn("--skip-first-lsearch-method", argv)
            self.assertIn("UNG__hybrid", argv)
            self.assertNotIn("UNG:Original UNG:#4c78a8:circle", argv)


    def test_config_shell_passes_per_method_query_task_and_lsearch(self):
        script_dir = Path(__file__).resolve().parent
        config_shell = script_dir / "plot_qps_recall_config.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_path = tmp_path / "argv.json"
            recorder = tmp_path / "record_argv.py"
            recorder.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"open({str(argv_path)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            )

            config = {
                "script": str(recorder),
                "dataset": "Reviews",
                "query_task": "query_selected_time_advantage",
                "result_task": None,
                "paths": {
                    "results_root": str(tmp_path / "results"),
                    "data_root": str(tmp_path / "data"),
                    "output_dir": str(tmp_path / "plot"),
                },
                "methods": [
                    {
                        "method_dir": "FAVOR",
                        "label": "FAVOR",
                        "color": "#b279a2",
                        "marker": "diamond",
                        "query_task": "query_selected_recall_advantage",
                        "lsearch": {
                            "start": 10000,
                            "step": 10000,
                            "end": 200000,
                        },
                    }
                ],
                "lsearch": {
                    "start": 1000,
                    "step": 1000,
                    "end": 20000,
                },
                "draw_options": {
                    "sort_mode": "time-column-only",
                    "legend_outside": True,
                    "skip_missing": False,
                    "write_log_scale_svg": True,
                    "threshold_step": 0.05,
                },
            }
            config_path = tmp_path / "plot_config.json"
            config_path.write_text(json.dumps(config))

            subprocess.run(["bash", str(config_shell), str(config_path)], check=True)

            argv = json.loads(argv_path.read_text())
            self.assertIn("--lsearch-start", argv)
            self.assertIn("1000", argv)
            self.assertIn(
                "FAVOR:FAVOR:#b279a2:diamond:query_selected_recall_advantage:10000:10000:200000",
                argv,
            )

    def test_config_shell_passes_per_method_result_task(self):
        script_dir = Path(__file__).resolve().parent
        config_shell = script_dir / "plot_qps_recall_config.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_path = tmp_path / "argv.json"
            recorder = tmp_path / "record_argv.py"
            recorder.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"open({str(argv_path)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            )

            config = {
                "script": str(recorder),
                "dataset": "Amazon",
                "query_task": "query_selected_recall_advantage",
                "result_task": None,
                "paths": {
                    "results_root": str(tmp_path / "results"),
                    "data_root": str(tmp_path / "data"),
                    "output_dir": str(tmp_path / "plot"),
                },
                "methods": [
                    {
                        "method_dir": "Curator",
                        "label": "Curator",
                        "color": "#9d755d",
                        "marker": "star",
                        "query_task": "query_selected_recall_advantage",
                        "result_task": "query_selected_recall_advantage_search_ef64_search_ef8192",
                    }
                ],
                "lsearch": {
                    "start": 1000,
                    "step": 1000,
                    "end": 20000,
                },
                "draw_options": {
                    "sort_mode": "time-column-only",
                    "legend_outside": True,
                    "skip_missing": False,
                    "write_log_scale_svg": True,
                    "threshold_step": 0.05,
                },
            }
            config_path = tmp_path / "plot_config.json"
            config_path.write_text(json.dumps(config))

            subprocess.run(["bash", str(config_shell), str(config_path)], check=True)

            argv = json.loads(argv_path.read_text())
            self.assertIn(
                "Curator:Curator:#9d755d:star:query_selected_recall_advantage:query_selected_recall_advantage_search_ef64_search_ef8192",
                argv,
            )

    def test_config_shell_omits_output_dir_when_json_output_dir_is_null(self):
        script_dir = Path(__file__).resolve().parent
        config_shell = script_dir / "plot_qps_recall_config.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_path = tmp_path / "argv.json"
            recorder = tmp_path / "record_argv.py"
            recorder.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"open({str(argv_path)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            )

            config = {
                "script": str(recorder),
                "dataset": "Reviews",
                "query_task": "query_selected_recall_advantage",
                "result_task": None,
                "paths": {
                    "results_root": str(tmp_path / "results"),
                    "data_root": str(tmp_path / "data"),
                    "output_dir": None,
                },
                "methods": [
                    {
                        "method_dir": "UNG__hybrid",
                        "label": "Original UNG",
                        "color": "#4c78a8",
                        "marker": "circle",
                    }
                ],
                "lsearch": {
                    "start": 1000,
                    "step": 1000,
                    "end": 20000,
                },
                "draw_options": {
                    "sort_mode": "time-column-only",
                    "legend_outside": True,
                    "skip_missing": False,
                    "write_log_scale_svg": True,
                    "threshold_step": 0.05,
                },
            }
            config_path = tmp_path / "plot_config.json"
            config_path.write_text(json.dumps(config))

            subprocess.run(["bash", str(config_shell), str(config_path)], check=True)

            argv = json.loads(argv_path.read_text())
            self.assertNotIn("--output-dir", argv)
            self.assertIn("--result-task", argv)
            self.assertIn("query_selected_recall_advantage_1000_1000_20000", argv)


if __name__ == "__main__":
    unittest.main()
