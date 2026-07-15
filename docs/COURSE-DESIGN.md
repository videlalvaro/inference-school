# Course Design

## Thesis

The motivation for a LeetCode-style problem here is direct: every problem
removes one piece of "magic" from inference and leaves behind a working part of
the final engine.

A general algorithm puzzle can feel detached from current engineering work,
especially when a coding agent can produce a plausible answer. In this course,
code generation is not the scarce skill. The scarce skills are:

- translating an equation into shapes, layouts, and execution steps;
- choosing what to compute, cache, quantize, or fuse;
- predicting whether an operator is limited by compute, memory, or overhead;
- validating numerical behavior against an independent implementation;
- reading a profile and deciding what change is justified;
- explaining why prefill and decode need different kernels and policies.

Coding agents are allowed. They can help with syntax, APIs, and alternative
implementations. A problem is complete only when the learner supplies the
prediction, measurements, and explanation, so generated code alone cannot
satisfy the evidence gate.

## One engine, not a puzzle collection

Every implementation follows this progression:

```text
equation -> scalar CPU oracle -> parallel CPU/Metal operator
         -> benchmark -> model component -> end-to-end engine
```

Later problems reuse earlier artifacts. Matrix-vector multiplication calls on
dot-product reasoning. Q/K/V projection reuses matrix multiplication. Attention
reuses Q/K/V, masking, softmax, and weighted sums. The decode loop reuses the
transformer block and adds state management through the KV cache.

This cumulative structure gives each exercise an immediate answer to "why am I
solving this?"

## Problem types

Each roadmap item has one primary type, although a lesson may combine them.

| Type | Main activity | Required evidence |
| --- | --- | --- |
| Math lab | Derive an operation and work a small example | Shapes, units, and a hand-computed fixture |
| Operator | Build a readable CPU implementation | Correctness and numerical edge cases |
| Kernel | Map an operator to Metal | Grid, memory spaces, synchronization, and parity |
| Systems lab | Design storage or scheduling behavior | Invariants, memory budget, and integration tests |
| Investigation | Compare valid implementations | Prediction, controlled benchmark, and interpretation |
| Capstone | Combine previous components | End-to-end correctness, profile, and engineering report |

## The tutorial standard

Every problem directory must contain one dedicated tutorial with these
sections. A short prompt plus a test file is not sufficient.

1. **Why this exists**: the exact place the operation appears in inference.
2. **Learning outcomes**: observable things the learner will be able to do.
3. **Prerequisites**: prior problem IDs and only the math needed now.
4. **Vocabulary**: new terms with operational definitions.
5. **Math from first principles**: notation, derivation, and a small numerical
   example.
6. **Shape and dtype contract**: dimensions, strides, precision, and errors.
7. **CPU reference path**: a scalar or otherwise readable implementation.
8. **Correctness method**: fixtures, tolerances, edge cases, and an independent
   oracle.
9. **Performance model**: FLOPs, bytes moved, arithmetic intensity, allocation,
   and dispatch costs.
10. **Metal mapping**: grids, threadgroups, SIMD groups, memory spaces, barriers,
    and bounds behavior.
11. **Implementation checkpoints**: small stages that can be checked separately.
12. **Experiments**: a controlled size or layout sweep with a written prediction.
13. **Engine integration**: where the artifact is used next.
14. **Tradeoff questions**: cases in which a different implementation wins.
15. **Hints and canonical solution**: separated from the main path.

## Completion contract

A learner completes a problem when all applicable items are present:

- The exercise passes its correctness judge.
- CPU and Metal outputs agree within the stated numerical tolerance.
- Error behavior for invalid shapes is tested.
- The benchmark is run in a release build with the machine and input size noted.
- A prediction was written before the benchmark.
- The result is interpreted in terms of compute, memory traffic, launch cost, or
  synchronization rather than just "faster" or "slower."
- The component is connected to the cumulative engine or a later problem names
  the exact integration point.
- The learner can explain the implementation without reading it line by line.

## Design constraints

### CPU before Metal

The CPU version is a semantic oracle, not a disposable warm-up. GPU debugging
becomes much easier when shape rules and numerical expectations are already
executable.

### Shared judges

Starter and canonical implementations run through the same judge. This prevents
the solution path from silently relying on easier fixtures.

### Explicit shape, layout, and dtype

Every API states all three. A tensor shape without its physical layout is
insufficient for kernel work, and an operation without its accumulation dtype is
insufficient for numerical reasoning.

### Benchmarks answer one question

A benchmark changes one independent variable at a time. End-to-end timing and
kernel-only timing are labeled separately. Warm-up, build configuration, input
shape, and synchronization points are part of the result.

### Optimization follows evidence

The course starts with the clearest correct implementation. Tiling, vectorized
loads, SIMD-group operations, fusion, and quantization are introduced only after
the baseline exposes a specific cost.

### Canonical answers are inspectable but separate

Answers live in `InferenceSchoolSolutions`; learner edits live in `InferenceSchoolExercises`.
The CLI checks learner code by default and checks the answer only with
`--solution`.

## Assessment in an agent-rich workflow

The useful division of labor is:

- The learner owns the model of the operation, the prediction, and the judgment.
- The compiler owns type and language correctness.
- Tests own reproducible behavioral checks.
- Profilers own timing and hardware-counter observations.
- Coding agents may suggest or implement code, but their output is treated as a
  hypothesis until the other evidence agrees.

This is closer to production inference engineering than a closed-book coding
exercise. The goal is not to prove that a person can type a reduction loop. It
is to make them capable of noticing when a reduction is numerically wrong,
mapped poorly to the GPU, or irrelevant to the actual bottleneck.
