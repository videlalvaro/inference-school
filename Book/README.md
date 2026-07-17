# Inference School companion book

This directory contains the reproducible PDF and EPUB publication pipeline for
Inference School companion book. The curriculum remains authoritative in `Problems/`,
and code listings are read directly from `Sources/InferenceSchoolExercises/` and
`Sources/InferenceSchoolSolutions/` when the manuscripts are generated.

## Build

From this directory:

```sh
make book
```

The final PDF and EPUB are copied to `dist/Inference-School-Companion.pdf` and
`dist/Inference-School-Companion.epub`. Generated manuscripts, LaTeX, SVG diagrams,
auxiliary files, and intermediate artifacts remain under `Book/build/`. Commit
both published artifacts when curriculum or book source changes affect them.

To build only the intermediate EPUB under `Book/build/`:

```sh
make epub
```

To rebuild and verify chapter, diagram, exercise, solution, callout, appendix,
published-copy, PDF text-layer, LaTeX-log, and EPUB package completeness:

```sh
make check
```

Verification compares the generated files with the tracked copies in `dist/`
without replacing them. Run `make book` explicitly to publish a rebuilt pair.

To compile only Chapter 001 while working on the publication style:

```sh
make sample
```

For a one-chapter EPUB styling loop:

```sh
make sample-epub
```

Requirements: Python 3, Pandoc, LuaLaTeX, and latexmk. `make check` also uses
Ghostscript for noninteractive PDF text extraction and EPUBCheck for strict
EPUB validation. No browser or Mermaid CLI is required; the generator
translates the course's flowchart subset into native TikZ for PDF and SVG for
EPUB.
