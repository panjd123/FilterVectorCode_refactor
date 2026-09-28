#!/usr/bin/env python3
"""Shared registry for complete Recall--QPS figures required by the paper."""

PAPER_RECALL_QPS_FIGURES = (
    ("principal_zero", "Principal 0L systems"),
    ("one_layer_topology", "1L LNG/Trie overlay topology"),
    ("representative_depth",
     "Representative 0L, 1L, and 2L indexes"),
    ("upper_authorization", "2L-LT routing ablation"),
)

PAPER_FAMILY_NAMES = tuple(name for name, _ in PAPER_RECALL_QPS_FIGURES)
