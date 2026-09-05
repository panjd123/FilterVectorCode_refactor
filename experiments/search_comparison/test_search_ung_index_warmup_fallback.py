#!/usr/bin/env python3

from pathlib import Path
import unittest


class SearchUngIndexWarmupFailureTest(unittest.TestCase):
    def test_gpu_cover_frontier_warmup_failure_exits_without_cpu_fallback(self):
        source = Path(__file__).resolve().parents[2] / "UNG/codes/apps/search_UNG_index.cpp"
        text = source.read_text(encoding="utf-8")

        warmup_start = text.index("Warming up gpu_cover_frontier provider")
        search_start = text.index("// init query stats")
        warmup_block = text[warmup_start:search_start]

        self.assertIn("GPU cover frontier warm-up failed", warmup_block)
        self.assertIn("return -1", warmup_block)
        self.assertNotIn("Continuing with cpu_min_super_sets", warmup_block)
        self.assertNotIn("entry_group_provider = ANNS::EntryGroupProviderImpl::CpuMinSuperSets", warmup_block)

    def test_gpu_cover_frontier_preflights_dense_descendant_memory(self):
        source = Path(__file__).resolve().parents[2] / "UNG/codes/src/ung_gpu_cover_frontier_provider.cu"
        text = source.read_text(encoding="utf-8")

        self.assertIn("cudaMemGetInfo", text)
        self.assertIn("requires dense descendant bitset", text)


if __name__ == "__main__":
    unittest.main()
