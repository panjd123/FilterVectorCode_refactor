# Multilevel UNG paper

`main.tex` is the submission-style English manuscript.  Quantitative claims
live in `generated_results.tex` so they can be regenerated only from validated
artifacts instead of being copied from development notes.

For an informal local check, build with an ACM-compatible TeX installation:

```bash
latexmk -pdf main.tex
```

The benchmark host has a pinned Tectonic binary outside `PATH`.  The
working static binary can be invoked directly as:

```bash
/home/sunyahui/.local/opt/tectonic-0.15.0-musl/tectonic \
  --keep-logs --keep-intermediates main.tex
```

Two evidence contracts are intentionally separate:

- `generate_authoritative_paper_results.py` is the publication gate for the
  complete formal and repeated-build protocol. It remains fail-closed.
- `generate_deadline_paper_results.py` accepts only the completed bounded
  campaign: three complete Amazon figure families, five held-out workloads,
  independent detailed profiles, and the explicit incomplete-build manifest.
  It labels query numbers as `1 cold + 2 warm` screen-level values and refuses
  to claim end-to-end GPU hierarchy speedup.

The deadline generator copies validated figures into `generated_figures/`,
writes `generated_results.tex` and the Chinese report atomically, and records
full input hashes in `runs/deadline_paper_20260928/manifest.json`.

The complete Amazon base-by-overlay topology factorial is generated separately
from same-search-binary and same-builder inputs. Its LaTeX table lives in
`generated_topology_factorial.tex`; the complete 14-by-9 equal-Recall matrix,
best configurations, controlled upper-level replacements, matched-provider
base comparisons, full-grid fixed-configuration ranking, and input manifest
live in `generated_data/`. Cross-base
ratios are system comparisons because each base uses its compatible entry
provider; only one-letter upper-level replacements hold the provider fixed.

`continue_after_build.py` also hashes the compiler and every direct paper
input, rejects unresolved references and BibTeX failures, and writes the final
PDF and provenance manifest.  That path, rather than an ad hoc local build,
defines the publication artifact.

Before treating the full-protocol draft as final:

1. Complete and validate screen, crossing, formal, and profile query passes.
2. Complete held-out automatic-versus-oracle experiments.
3. Complete repeated end-to-end CPU/GPU construction and quality checks.
4. Replace every `\pending{...}` marker through the result generator.
5. Compile the PDF and inspect table/algorithm placement.
