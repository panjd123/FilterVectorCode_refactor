# Multilevel UNG paper

`main.tex` is the submission-style English manuscript.  Quantitative claims
live in `generated_results.tex` so they can be regenerated only from validated
artifacts instead of being copied from development notes.

Build with an ACM-compatible TeX installation:

```bash
latexmk -pdf main.tex
```

The remote benchmark host currently has neither `latexmk` nor `pdflatex` in
`PATH`; source-level checks therefore run there, while PDF compilation requires
a TeX environment.

Before treating the draft as final:

1. Complete and validate screen, crossing, formal, and profile query passes.
2. Complete held-out automatic-versus-oracle experiments.
3. Complete repeated end-to-end CPU/GPU construction and quality checks.
4. Replace every `\pending{...}` marker through the result generator.
5. Compile the PDF and inspect table/algorithm placement.
