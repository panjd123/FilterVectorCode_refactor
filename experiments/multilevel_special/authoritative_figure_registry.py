#!/usr/bin/env python3
"""Shared registry for complete Recall--QPS figures required by the paper."""

PAPER_RECALL_QPS_FIGURES = (
    ("principal_zero", "Principal zero-layer systems"),
    ("one_layer_topology", "One-layer LNG/Trie topology"),
    ("representative_depth",
     "Representative zero-, one-, and two-layer indexes"),
    ("upper_authorization", "Exact upper-authorization routing"),
)

PAPER_FAMILY_NAMES = tuple(name for name, _ in PAPER_RECALL_QPS_FIGURES)
