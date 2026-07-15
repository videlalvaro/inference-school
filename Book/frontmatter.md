# Preface {-}

Inference School builds one small decoder-only transformer inference engine from first
principles. The chapters begin with a scalar reduction and end with a measured,
inspectable generation pipeline. Swift provides the readable semantic path;
Metal exposes the memory, synchronization, and execution decisions that make
inference a systems problem.

This companion is generated from the same lesson and source files as the native
Inference School Studio application. It is not a parallel curriculum. Equations,
exercises, diagrams, starter files, experiments, and canonical implementations
remain tied to the repository that executes them.

## How to use this book {-}

Each chapter follows one loop:

1. Work through the derivation and the small numerical example.
2. Write down the requested prediction before running code.
3. Implement the Swift CPU path and make it pass its focused judge.
4. Implement the Metal path where the chapter defines one.
5. Run the controlled experiments in a release build.
6. Explain the result in terms of shapes, bytes, operations, and execution
   boundaries.
7. Only then read the worked solution at the end of the chapter.

The worked solutions are intentionally more than source dumps. Each ending
states the strategy, identifies the invariants and failure modes, maps the
important code regions to the ideas in the chapter, and then includes the full
chapter-owned canonical files. Infrastructure established in earlier chapters
is referenced rather than repeatedly relisted.

Two generated appendices make the volume self-contained: a one-token decode
walkthrough names the complete inference path, and a just-in-time math primer
collects the notation and derivations used across the exercises.

\begin{chapterfocus}
A green judge is evidence of contract compliance, not the end of the exercise.
The durable artifact is the explanation: why the implementation is correct,
which costs it pays, and under which shapes or machine conditions another design
would be preferable.
\end{chapterfocus}

## Build and notation {-}

Commands assume an Apple Silicon Mac with the Swift and Metal command-line tools
installed. Correctness runs use debug builds unless stated otherwise;
measurements use release builds. Tensor shapes are written in logical order,
and every performance claim should identify its units and measurement boundary.

The PDF is vector-first. Equations are typeset by LaTeX and curriculum
flowcharts are translated to TikZ, so both remain sharp under magnification.
