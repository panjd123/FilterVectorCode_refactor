#!/usr/bin/env python3

import unittest

import prepare_deadline_evidence_campaign as deadline


class DeadlineCpuBackendTest(unittest.TestCase):
    def test_force_cpu_backend_disables_both_gpu_routes(self) -> None:
        case = {
            "name": "example",
            "benchmark_profile": "hybrid_gpu_intra",
            "env": {
                "UNG_SPECIAL_BLOCK_GPU_INTRA": "1",
                "UNG_SPECIAL_BLOCK_GPU_INTER": "1",
                "UNG_SPECIAL_INTRA_ROUTE": "1",
                "UNCHANGED": "value",
            },
        }

        deadline.force_cpu_backend(case)
        deadline.validate_cpu_backend(case)

        self.assertEqual(case["benchmark_profile"], "cpu")
        self.assertEqual(case["env"]["UNCHANGED"], "value")
        self.assertEqual(
            {key: case["env"][key] for key in deadline.CPU_BACKEND_ENV},
            deadline.CPU_BACKEND_ENV,
        )

    def test_validate_cpu_backend_rejects_size_routed_cuda(self) -> None:
        case = {
            "name": "example",
            "benchmark_profile": "cpu",
            "env": dict(deadline.CPU_BACKEND_ENV),
        }
        case["env"]["UNG_SPECIAL_INTRA_ROUTE"] = "1"

        with self.assertRaisesRegex(ValueError, "active GPU route"):
            deadline.validate_cpu_backend(case)


if __name__ == "__main__":
    unittest.main()
