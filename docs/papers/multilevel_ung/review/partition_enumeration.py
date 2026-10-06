"""Finite structural enumeration; no vector data, distance work, or benchmarks."""
from itertools import product
import json


def partition(parents, weights, threshold):
    n = len(parents)
    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parents[i]].append(i)
    residual = [frozenset() for _ in range(n)]
    counts = [0] * n
    blocks = {}
    for v in reversed(range(n)):
        pending = set([v] if weights[v] else [])
        count = 0
        for c in children[v]:
            pending.update(residual[c])
            count += counts[c]
        if v and sum(weights[x] for x in pending) > threshold:
            blocks[v] = frozenset(pending)
            pending.clear()
            count += 1
        residual[v] = frozenset(pending)
        counts[v] = count
    return blocks, residual, counts


checked = comparisons = 0
equal_count_nonnested = None
for n in range(1, 6):
    for p in product(*(range(i) for i in range(1, n + 1))):
        parents = (0,) + p
        for w in product(range(3), repeat=n):
            weights = (0,) + w
            total = sum(weights)
            if total == 0:
                continue
            checked += 1
            previous = None
            for threshold in range(total + 1):
                blocks, residual, counts = partition(parents, weights, threshold)
                assert len(blocks) <= total // (threshold + 1)
                assert sum(sum(weights[x] for x in b) for b in blocks.values()) + sum(weights[x] for x in residual[0]) == total
                if previous is not None:
                    old_blocks, old_residual, old_counts = previous
                    for v in range(n + 1):
                        comparisons += 1
                        assert old_counts[v] >= counts[v]
                        if old_counts[v] == counts[v]:
                            assert old_residual[v] >= residual[v]
                    if len(old_blocks) == len(blocks) and equal_count_nonnested is None:
                        if any(not any(b <= c for c in blocks.values()) for b in old_blocks.values()):
                            equal_count_nonnested = dict(parents=parents, weights=weights,
                                thresholds=[threshold - 1, threshold],
                                low={k: sorted(v) for k, v in old_blocks.items()},
                                high={k: sorted(v) for k, v in blocks.items()})
                previous = blocks, residual, counts

examples = {}
for name, parents, weights, thresholds in [
    ('nonnested_decrease', (0,0,1,2,1), (0,1,2,3,2), (2,4)),
    ('coverage_increase', (0,0,1,1), (0,1,3,3), (2,6)),
    ('lng_retag', (0,0,1,0,3), (0,1,1,0,1), (1,)),
]:
    examples[name] = []
    for threshold in thresholds:
        blocks, residual, _ = partition(parents, weights, threshold)
        examples[name].append(dict(threshold=threshold,
            blocks={k: sorted(v) for k,v in blocks.items()},
            represented_mass=sum(sum(weights[x] for x in b) for b in blocks.values()),
            residual=sorted(residual[0])))

print(json.dumps(dict(weighted_trees_checked=checked,
    subtree_threshold_comparisons=comparisons,
    equal_count_nonnested=equal_count_nonnested,
    examples=examples), indent=2))
