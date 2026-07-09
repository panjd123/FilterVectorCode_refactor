#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Node:
    label: int
    parent: int
    children: Dict[int, int] = field(default_factory=dict)
    group_size: int = 0
    depth: int = 0
    labels: Tuple[int, ...] = ()


@dataclass
class Block:
    block_id: int
    node_id: int
    root_labels: Tuple[int, ...]
    root_group: int
    point_count: int = 0
    subtree_point_count: int = 0
    member_groups: List[int] = field(default_factory=list)
    child_blocks: List[int] = field(default_factory=list)

    @property
    def is_trivial(self) -> bool:
        return self.root_group >= 0 and len(self.member_groups) == 1 and self.member_groups[0] == self.root_group and not self.child_blocks


def parse_counts(path: Path) -> Tuple[int, Counter[Tuple[int, ...]]]:
    counts: Counter[Tuple[int, ...]] = Counter()
    total = 0
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            labs = tuple(sorted(int(x) for x in s.replace(",", " ").split()))
            counts[labs] += 1
            total += 1
    return total, counts


def parse_queries(path: Path) -> List[Tuple[int, ...]]:
    queries = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s:
                queries.append(tuple(sorted(int(x) for x in s.replace(",", " ").split())))
    return queries


def build_trie(counts: Counter[Tuple[int, ...]]):
    nodes = [Node(-1, -1, labels=(), depth=0)]
    group_labels: List[Tuple[int, ...]] = []
    group_sizes: List[int] = []
    terminal_node_to_group: Dict[int, int] = {}
    for labs, cnt in sorted(counts.items()):
        cur = 0
        prefix: List[int] = []
        for lab in labs:
            prefix.append(lab)
            nxt = nodes[cur].children.get(lab)
            if nxt is None:
                nxt = len(nodes)
                nodes[cur].children[lab] = nxt
                nodes.append(Node(lab, cur, labels=tuple(prefix), depth=nodes[cur].depth + 1))
            cur = nxt
        nodes[cur].group_size = cnt
        terminal_node_to_group[cur] = len(group_labels)
        group_labels.append(labs)
        group_sizes.append(cnt)
    return nodes, group_labels, group_sizes, terminal_node_to_group


def compute_subtree_points(nodes: List[Node]):
    subp = [0] * len(nodes)
    subg = [0] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        p = nodes[i].group_size
        g = 1 if nodes[i].group_size > 0 else 0
        for child in nodes[i].children.values():
            p += subp[child]
            g += subg[child]
        subp[i] = p
        subg[i] = g
    return subp, subg


def collect_block_members(nodes: List[Node], root: int, node_to_block: List[int], terminal_node_to_group: Dict[int, int], block: Block):
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur != root and node_to_block[cur] != 0:
            block.child_blocks.append(node_to_block[cur])
            continue
        group = terminal_node_to_group.get(cur)
        if group is not None:
            block.member_groups.append(group)
            block.point_count += nodes[cur].group_size
        stack.extend(nodes[cur].children.values())
    block.member_groups.sort()
    block.child_blocks = sorted(set(block.child_blocks))


def build_special_blocks(nodes: List[Node], subp: List[int], threshold: int, terminal_node_to_group: Dict[int, int]) -> List[Block]:
    uncovered = [0] * len(nodes)
    node_to_block = [0] * len(nodes)
    blocks: List[Block] = []
    for idx in range(len(nodes) - 1, -1, -1):
        u = nodes[idx].group_size
        for child in nodes[idx].children.values():
            u += uncovered[child]
        uncovered[idx] = u
        if idx == 0:
            continue
        if u > threshold:
            block = Block(
                block_id=len(blocks) + 1,
                node_id=idx,
                root_labels=nodes[idx].labels,
                root_group=terminal_node_to_group.get(idx, -1),
                subtree_point_count=subp[idx],
            )
            collect_block_members(nodes, idx, node_to_block, terminal_node_to_group, block)
            blocks.append(block)
            node_to_block[idx] = block.block_id
            uncovered[idx] = 0
    return blocks


def invert_labels(labels: List[Tuple[int, ...]]):
    inv: Dict[int, List[int]] = defaultdict(list)
    for idx, labs in enumerate(labels):
        for lab in labs:
            inv[lab].append(idx)
    return inv


def match_ids(query: Tuple[int, ...], inv: Dict[int, List[int]]):
    lists = []
    for lab in query:
        if lab not in inv:
            return []
        lists.append(inv[lab])
    lists.sort(key=len)
    cur = set(lists[0])
    for lst in lists[1:]:
        cur.intersection_update(lst)
        if not cur:
            break
    return list(cur)


def pct(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    return float(xs[lo]) if lo == hi else xs[lo] * (hi - k) + xs[hi] * (k - lo)


def path(nodes: List[Node], i: int) -> str:
    return " ".join(map(str, nodes[i].labels))


def summarize(nodes: List[Node], group_labels: List[Tuple[int, ...]], group_sizes: List[int], blocks: List[Block], queries: List[Tuple[int, ...]]):
    group_inv = invert_labels(group_labels)
    block_inv = invert_labels([b.root_labels for b in blocks])
    block_points = [b.point_count for b in blocks]
    block_groups = [len(b.member_groups) for b in blocks]
    trivial = [b.is_trivial for b in blocks]

    total_matched = 0
    total_special = 0
    total_nontrivial = 0
    per_rows = []
    for qi, query in enumerate(queries):
        gids = match_ids(query, group_inv)
        matched = sum(group_sizes[g] for g in gids)
        bidxs = match_ids(query, block_inv)
        special = sum(block_points[i] for i in bidxs)
        nontrivial = sum(block_points[i] for i in bidxs if not trivial[i])
        total_matched += matched
        total_special += special
        total_nontrivial += nontrivial
        per_rows.append(
            (
                qi,
                " ".join(map(str, query)),
                matched,
                special,
                nontrivial,
                matched - special,
                special / matched if matched else 0.0,
                nontrivial / matched if matched else 0.0,
                len(bidxs),
            )
        )

    nontrivial_points = sum(b.point_count for b in blocks if not b.is_trivial)
    nontrivial_groups = sum(len(b.member_groups) for b in blocks if not b.is_trivial)
    top_blocks = sorted(blocks, key=lambda b: b.point_count, reverse=True)[:20]
    return {
        "block_count": len(blocks),
        "trivial_blocks": sum(1 for b in blocks if b.is_trivial),
        "nontrivial_blocks": sum(1 for b in blocks if not b.is_trivial),
        "covered_points": sum(block_points),
        "covered_groups": sum(block_groups),
        "nontrivial_points": nontrivial_points,
        "nontrivial_groups": nontrivial_groups,
        "total_matched": total_matched,
        "total_special": total_special,
        "total_nontrivial": total_nontrivial,
        "total_outside": total_matched - total_special,
        "point_p50": pct(block_points, 50),
        "point_p90": pct(block_points, 90),
        "point_p95": pct(block_points, 95),
        "point_p99": pct(block_points, 99),
        "point_max": max(block_points) if block_points else 0,
        "group_p50": pct(block_groups, 50),
        "group_p90": pct(block_groups, 90),
        "group_p95": pct(block_groups, 95),
        "group_p99": pct(block_groups, 99),
        "group_max": max(block_groups) if block_groups else 0,
        "per_rows": per_rows,
        "top_blocks": top_blocks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-label-file", type=Path, required=True)
    ap.add_argument("--query-label-file", type=Path, required=True)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--data-note", default="Amazon 100% x1/original label file; do not use repeated xN group sizes for main claims.")
    args = ap.parse_args()

    total, counts = parse_counts(args.base_label_file)
    queries = parse_queries(args.query_label_file)
    nodes, group_labels, group_sizes, terminal_node_to_group = build_trie(counts)
    subp, _ = compute_subtree_points(nodes)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for threshold in args.thresholds:
        blocks = build_special_blocks(nodes, subp, threshold, terminal_node_to_group)
        summary = summarize(nodes, group_labels, group_sizes, blocks, queries)
        rows.append((threshold, blocks, summary))

        csv_path = args.out_prefix.with_name(f"{args.out_prefix.name}_T{threshold}.per_query.csv")
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "query_id",
                    "labels",
                    "matched_points",
                    "query_special_points",
                    "query_nontrivial_special_points",
                    "outside_points",
                    "query_special_ratio",
                    "query_nontrivial_special_ratio",
                    "query_special_block_count",
                ]
            )
            w.writerows(summary["per_rows"])

        blocks_csv = args.out_prefix.with_name(f"{args.out_prefix.name}_T{threshold}.blocks.csv")
        with blocks_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block_id", "root_labels", "root_group", "point_count", "subtree_point_count", "member_group_count", "child_block_count", "is_trivial"])
            for b in blocks:
                w.writerow([b.block_id, " ".join(map(str, b.root_labels)), b.root_group, b.point_count, b.subtree_point_count, len(b.member_groups), len(b.child_blocks), int(b.is_trivial)])

    md = args.out_prefix.with_suffix(".md")
    with md.open("w") as f:
        f.write("# Special Block Threshold Sweep\n\n")
        f.write("口径：bottom-up uncovered_points 定义；当 `uncovered_points(node) > threshold` 时建立特异块。旧 tiny_group_size/tiny_group_ratio 和 frontier 定义均不使用。\n\n")
        f.write(f"base_label_file: `{args.base_label_file}`\n\n")
        f.write(f"query_label_file: `{args.query_label_file}`\n\n")
        f.write(f"data_note: {args.data_note}\n\n")
        f.write(f"total_points={total}, total_groups={len(counts)}, queries={len(queries)}\n\n")
        f.write("查询覆盖率按所有 query 的 matched points 直接加和，不平均单 query 比例。\n\n")
        f.write("## Summary\n\n")
        f.write("| threshold | blocks | trivial_blocks | nontrivial_blocks | tree_special_points | tree_special_ratio | tree_nontrivial_points | tree_nontrivial_ratio | tree_special_groups | total_query_matched | query_special_points | query_special_ratio | query_nontrivial_points | query_nontrivial_ratio | query_outside_points | query_outside_ratio | block_point_p50 | block_point_p90 | block_point_p99 | block_point_max | block_group_p50 | block_group_p90 | block_group_p99 | block_group_max |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for threshold, _blocks, s in rows:
            total_matched = s["total_matched"]
            f.write(
                f"| {threshold} | {s['block_count']} | {s['trivial_blocks']} | {s['nontrivial_blocks']} | "
                f"{s['covered_points']} | {s['covered_points']/total if total else 0:.6f} | "
                f"{s['nontrivial_points']} | {s['nontrivial_points']/total if total else 0:.6f} | "
                f"{s['covered_groups']} | {total_matched} | {s['total_special']} | {s['total_special']/total_matched if total_matched else 0:.6f} | "
                f"{s['total_nontrivial']} | {s['total_nontrivial']/total_matched if total_matched else 0:.6f} | "
                f"{s['total_outside']} | {s['total_outside']/total_matched if total_matched else 0:.6f} | "
                f"{s['point_p50']:.1f} | {s['point_p90']:.1f} | {s['point_p99']:.1f} | {s['point_max']} | "
                f"{s['group_p50']:.1f} | {s['group_p90']:.1f} | {s['group_p99']:.1f} | {s['group_max']} |\n"
            )
        for threshold, _blocks, s in rows:
            f.write(f"\n## Top Blocks, threshold={threshold}\n\n")
            f.write("| rank | block_id | points | groups | child_blocks | trivial | root_labels |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---|\n")
            for rank, block in enumerate(s["top_blocks"][:10], 1):
                f.write(f"| {rank} | {block.block_id} | {block.point_count} | {len(block.member_groups)} | {len(block.child_blocks)} | {int(block.is_trivial)} | `{path(nodes, block.node_id)}` |\n")

    print(md)


if __name__ == "__main__":
    main()
