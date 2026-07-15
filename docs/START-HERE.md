# Start Here

Inference School is a hands-on course for building a small decoder-only language-model
inference engine from individual Swift and Metal operators. The primary
learning environment is the native Inference School Studio app. The same lessons and
checks also remain available from the command line.

Begin with [Problem 000: Start Here](../Problems/000-start-here/README.md). It
defines the learning workflow and introduces tokens, embeddings, the residual
stream, Q/K/V, attention, logits, prefill, decode, and the KV cache. No prior
transformer or Metal knowledge is assumed.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer
- Xcode or the Xcode command-line tools with Swift and Metal

Confirm the toolchain from the repository root:

```sh
swift --version
xcrun --find metal
```

## Launch Inference School Studio

Package the debug app and open it:

```sh
scripts/package-studio.sh debug
open "dist/Inference School Studio.app"
```

The package script builds the Studio and its separate runner, then bundles the
course sources under `dist/Inference School Studio.app`. Re-run it after changing app
code or bundled course content. The generated `dist/` directory is local build
output and should not be committed.

The Studio opens at lesson `000`. Its left column is the searchable lesson
catalog. Runnable lessons show a reader plus a workbench containing the
chapter-owned starter files and CPU or Metal checks. Lesson `000` is
reader-only, so it intentionally shows no editor or Run button.

The first runnable lesson asks you to choose or create a dedicated build folder
through the macOS folder picker. The Studio stores editable course sources,
compiler output, and generated learner executables under `Inference School Workspaces`
inside that folder. The saved macOS permission is reused on later launches and
can be replaced or forgotten from the Build Folder toolbar menu.

Use the Text Size toolbar menu to scale lesson text, controls, source code, and
results from 80% to 200%. Command-Plus increases the size, Command-Minus
decreases it, and Command-Zero restores 100%. The chosen size and lesson
checklists persist locally.

## Local execution and trust

The packaged Studio, its runner, and generated learner executable are signed to
use App Sandbox on macOS 15 and later. The Studio requests read, write, and
executable access only for the build folder you select. Its host signature also
contains the client entitlement required by WebKit; lesson, diagram, and editor
assets remain bundled, and built-in checks do not upload learner code to a
remote model or service. The packaged runner currently inherits the Studio
sandbox, including that entitlement.

This boundary does not make arbitrary code harmless. Learner code can read,
change, or delete files inside the selected build folder and can consume CPU,
memory, or GPU time until the runner's timeout or cancellation takes effect.
Use a dedicated folder and review third-party source before running it. The
command-line interface is not App Sandbox constrained; `swift run` checks use
the permissions of the terminal process that launches them.

## Command-line first ten minutes

The command line uses the same course sources and judges. Verify the harness:

```sh
swift run inference-school learn 001
swift run inference-school check 001 --solution
```

The solution check should report `5/5` for CPU and Metal. It verifies the course
harness; it does not complete your exercise. After reading lesson `000`, use
[Anatomy of One Token](ONE-TOKEN.md) as a compact map showing where the course's
operators fit in one decode step.

## One learning session

Open [Problem 001](../Problems/001-vector-dot/README.md) and work through it in
order. A session follows this loop:

### 1. Understand

Read "Why this exists," the learning outcomes, and the math. Compute the small
example by hand. Before running a benchmark, write down the requested
performance prediction.

### 2. Implement the readable version

Edit
[P001VectorDotExercise.swift](../Sources/InferenceSchoolExercises/P001VectorDotExercise.swift),
then run:

```sh
swift run inference-school check 001 --cpu
```

Keep working until it reports `CPU: 5/5 cases`. This CPU implementation is the
meaning of the operation against which the GPU version is judged.

### 3. Map it to the GPU

Read the Metal mapping and barrier sections in the tutorial. Edit
[P001VectorDot.metal](../Sources/InferenceSchoolExercises/Metal/P001VectorDot.metal),
then run:

```sh
swift run inference-school check 001 --metal
```

Do not stop at a passing kernel. Draw the reduction tree and account for the
bounds check, memory spaces, and every barrier.

### 4. Measure the tradeoff

Run the tutorial's size sweep in a release build. For example:

```sh
swift run -c release inference-school benchmark 001 --size 64 --iterations 100
swift run -c release inference-school benchmark 001 --size 4096 --iterations 100
swift run -c release inference-school benchmark 001 --size 1048576 --iterations 20
```

Compare the results with your prediction. Explain the crossover using fixed GPU
overhead, bytes moved, and arithmetic intensity. The explanation is part of the
exercise; a green judge alone is not completion.

### 5. Connect it to inference

Finish the tutorial's tradeoff questions and engine-integration section. You
should be able to point to the future GEMV and attention equations that reuse
the reduction you just built.

## What is available now

Problems 001 through 047 are complete runnable lessons. Together they cover
dense linear algebra, neural operators, embeddings, RoPE, materialized and
streaming attention, MHA/MQA/GQA, tiled fused attention, local attention, and
contiguous, ring, paged, and quantized KV caches. The weight-quantization module
adds symmetric INT8, groupwise scales, a byte-exact Q4 format, staged and fused
Q4 GEMV, and layerwise error diagnosis. The first assembly lessons add an
ordered decoder block, a bounds-checked educational weight container, byte-level
BPE, deterministic logits sampling, shared prefill/decode model state, an
executable buffer-lifetime planner, and named-capture parity over the educational
mini-model. The final module adds fused RMSNorm/QKV, stage-separated profiling,
static and continuous batching, probability-correct speculative decoding, and a
connected educational capstone with a real Metal parity slice.
Discover their paths and checks with:

```sh
swift run inference-school list
swift run inference-school learn 002
swift run inference-school show 006
swift run inference-school learn 047
```

Run the final evidence commands in a release build:

```sh
swift run -c release inference-school benchmark 043 --tokens 32 --iterations 20
swift run -c release inference-school profile 044 --prompt-tokens 16 --trials 7
swift run -c release inference-school capstone --prompt "ab c." --max-tokens 4
```

The capstone model is a seven-token deterministic educational fixture, not a
pretrained language model. Generation is CPU reference execution; Metal runs
only the fused-QKV and RoPE verification slice named in its report. The native
Studio and command line are two interfaces over the same course and judges;
there is no hosted web platform.