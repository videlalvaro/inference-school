# Inference School companion book

This directory contains the reproducible LaTeX publication pipeline for the
Inference School companion book. The curriculum remains authoritative in `Problems/`,
and code listings are read directly from `Sources/InferenceSchoolExercises/` and
`Sources/InferenceSchoolSolutions/` when the manuscript is generated.

## Build

From this directory:

```sh
make book
```

The final PDF is copied to the repository's `dist/Inference-School-Companion.pdf`.
Generated Markdown, LaTeX, auxiliary files, and the intermediate PDF remain
under `Book/build/`. Commit the published PDF when curriculum or book source
changes affect its contents.

To rebuild and verify chapter, diagram, exercise, solution, callout, appendix,
PDF-copy, text-layer, and LaTeX-log completeness:

```sh
make check
```

To compile only Chapter 001 while working on the publication style:

```sh
make sample
```

Requirements: Python 3, Pandoc, LuaLaTeX, and latexmk. `make check` also uses
Ghostscript for noninteractive PDF text extraction. No browser or Mermaid CLI
is required; the generator translates the flowchart subset used by the course
into native TikZ figures.
