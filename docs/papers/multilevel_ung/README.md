# Multilevel UNG paper

`main.tex` is the submission-style English manuscript.  Quantitative claims
live in `generated_results.tex` so they can be regenerated only from validated
artifacts instead of being copied from development notes.

For an informal local check, build with an ACM-compatible TeX installation:

```bash
latexmk -pdf main.tex
```

The benchmark host has a pinned Tectonic binary outside `PATH`.  The
authoritative finalizer invokes it directly as:

```bash
/home/sunyahui/.local/opt/tectonic-0.15.0-musl/tectonic -X compile \
  --keep-logs --keep-intermediates --outdir <paper-output> main.tex
```

`continue_after_build.py` also hashes the compiler and every direct paper
input, rejects unresolved references and BibTeX failures, and writes the final
PDF and provenance manifest.  That path, rather than an ad hoc local build,
defines the publication artifact.

Before treating the draft as final:

1. Complete and validate screen, crossing, formal, and profile query passes.
2. Complete held-out automatic-versus-oracle experiments.
3. Complete repeated end-to-end CPU/GPU construction and quality checks.
4. Replace every `\pending{...}` marker through the result generator.
5. Compile the PDF and inspect table/algorithm placement.
