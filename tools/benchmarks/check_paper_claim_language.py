#!/usr/bin/env python3
"""Lint paper-facing docs for over-strong or unsupported claim language.

This checker complements audit_paper_artifacts.py. Artifact audit verifies that
evidence files exist; this script scans manuscript text for phrases that would
overstate the current evidence, such as "lossless replacement" or "x400 solved".

The checker is intentionally conservative and small. It allows the same phrase
when the sentence is explicitly negated, e.g. "we do not claim lossless
replacement" or "不能写成无损替代".
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


def discover_default_files() -> tuple[Path, ...]:
    """Return reviewer-facing Markdown docs covered by the claim-language gate."""
    roots = (Path("README.md"), Path("docs"), Path("scripts/benchmarks"))
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".md":
            paths.append(root)
        elif root.is_dir():
            for path in root.rglob("*.md"):
                if "archive" in path.parts:
                    continue
                paths.append(path)
    return tuple(sorted(dict.fromkeys(paths), key=lambda p: str(p)))


DEFAULT_FILES = discover_default_files()


NEGATION_RE = re.compile(
    r"(不能|不可|不应|不是|不会|尚未|未|待验证|反例|negative|limitation|pending|not|do not|does not|"
    r"不写|不要声称|不能声称|不代表|不证明|不支持|阻力|边界|攻击|质疑|追问|回答 reviewer|"
    r"写成|仍需|仍需要|需要.*A/B|需要.*验证|"
    r"smoke|ablation|口径|历史点|推进到|should not|rather than|instead of|no measured|not yet)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    message: str


RULES = (
    Rule(
        name="lossless_or_universal_replacement",
        pattern=re.compile(
            r"(无损[^。\n]{0,24}替代|普遍[^。\n]{0,24}替代|lossless[^.\n]{0,40}replace|"
            r"universally?[^.\n]{0,40}replace|unconditional[^.\n]{0,40}replace)",
            re.IGNORECASE,
        ),
        message="FastGrnndCuda must be described as a speed/quality tradeoff, not a lossless universal replacement.",
    ),
    Rule(
        name="x400_solved",
        pattern=re.compile(r"(x400[^。\n.]{0,40}(已解决|solved)|解决[^。\n.]{0,20}x400)", re.IGNORECASE),
        message="x400 repair/reverse-tail is pending; do not describe x400 as solved.",
    ),
    Rule(
        name="router_solved",
        pattern=re.compile(
            r"((mixed exact/GNN|router|UNG_FAST_GRNND_BATCH_EXACT_NX)[^。\n.]{0,60}(solves?|已解决|解决)|"
            r"(solves?|已解决|解决)[^。\n.]{0,60}(mixed exact/GNN|router|UNG_FAST_GRNND_BATCH_EXACT_NX))",
            re.IGNORECASE,
        ),
        message="Mixed exact/GNN router has no measured result yet; keep it pending unless citing completed A/B.",
    ),
    Rule(
        name="custom_kernel_dominates_libraries",
        pattern=re.compile(
            r"((custom|自研|ours|我们的)[^。\n.]{0,50}(dominates?|beats all|普遍快于|全面超过)[^。\n.]{0,50}(cuBLAS|cuVS|SGEMM)|"
            r"(cuBLAS|cuVS|SGEMM)[^。\n.]{0,50}(dominated|被.*普遍.*超过))",
            re.IGNORECASE,
        ),
        message="The defensible claim is workload-level fusion, not universal dominance over cuBLAS/cuVS.",
    ),
    Rule(
        name="skip_additional_edges_full_quality",
        pattern=re.compile(
            r"(skip[^.\n]{0,30}additional[^.\n]{0,50}full[- ]quality|"
            r"跳过[^。\n]{0,20}additional[^。\n]{0,50}(完整|质量|主表))",
            re.IGNORECASE,
        ),
        message="skip additional_edges is only an ablation; full-quality claims require additional_edges enabled.",
    ),
)


def split_sentences(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    line_no = 1
    negative_example_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            line_no += 1
            continue
        if stripped.startswith("#"):
            negative_example_block = False
            line_no += 1
            continue
        if stripped.startswith("|") or stripped.startswith(">"):
            line_no += 1
            continue
        if re.search(r"(避免写|不能主张|当前论文不能主张|不要写|Do not claim)", stripped, re.IGNORECASE):
            negative_example_block = True
            line_no += 1
            continue
        if negative_example_block and stripped.startswith(("-", "*")):
            line_no += 1
            continue
        if negative_example_block and not stripped.startswith(("-", "*")):
            negative_example_block = False
        parts = re.split(r"(?<=[。.!?])\s+", stripped)
        for part in parts:
            if part:
                out.append((line_no, part))
        line_no += 1
    return out


def is_allowed(sentence: str) -> bool:
    return bool(NEGATION_RE.search(sentence))


def lint_file(path: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    text = path.read_text(errors="replace")
    for line_no, sentence in split_sentences(text):
        for rule in RULES:
            if rule.pattern.search(sentence) and not is_allowed(sentence):
                findings.append(
                    {
                        "file": str(path),
                        "line": str(line_no),
                        "rule": rule.name,
                        "message": rule.message,
                        "text": sentence,
                    }
                )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="Files to scan. Defaults to reviewer-facing Markdown docs outside docs/archive.")
    parser.add_argument("--allow-empty", action="store_true", help="Do not fail if a requested file is missing.")
    args = parser.parse_args()

    files = tuple(args.files) if args.files else DEFAULT_FILES
    missing = [p for p in files if not p.exists()]
    if missing and not args.allow_empty:
        for path in missing:
            print(f"[MISSING] {path}")
        raise SystemExit(2)

    findings: list[dict[str, str]] = []
    for path in files:
        if path.exists():
            findings.extend(lint_file(path))

    if findings:
        for item in findings:
            print(
                f"{item['file']}:{item['line']}: {item['rule']}: {item['message']}\n"
                f"  {item['text']}"
            )
        raise SystemExit(1)

    print(f"[OK] checked {len(files) - len(missing)} file(s); no over-strong claim language found.")


if __name__ == "__main__":
    main()
