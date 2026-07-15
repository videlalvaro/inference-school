# Anatomy of One Generated Token

This reading names the entire machine before the course builds each part. The
goal is to make every later problem locatable in one execution trace.

A large language model receives **tokens**, integer IDs representing pieces of
text or special markers, and predicts one next token at a time. **Inference** is
the act of running the model's already learned weights to make those
predictions. For a slower introduction to this vocabulary and the intended
learning workflow, begin with
[Start Here](../Problems/000-start-here/README.md).

## Symbols

| Symbol | Meaning |
| --- | --- |
| $S$ | Prompt sequence length |
| $T$ | Tokens currently held in the KV cache |
| $D$ | Model or residual-stream width |
| $L$ | Number of transformer layers |
| $H_q$ | Number of query heads |
| $H_{kv}$ | Number of key/value heads |
| $d_h$ | Width of one attention head |
| $D_{ff}$ | Hidden width of the feed-forward network |
| $V$ | Vocabulary size |

**Scope:** every example below uses batch size one: one generation request at a
time. Batching is introduced in Problem 045, after the single-request memory
and execution behavior is visible.

## Inference has two distinct stages

**Prefill** processes the complete input prompt together. **Decode** then
processes one newly generated token per step, because each step depends on the
token selected by the preceding step. Prefill exposes parallel work across
prompt positions; decode is serial across generated tokens and repeatedly
reuses model weights and saved context. Their different shapes lead to
different performance bottlenecks and often different kernels.

### Prefill

Prefill processes the prompt tokens. Its important shapes include a sequence
axis:

```text
token IDs                   [S]
embeddings                  [S, D]
Q                           [S, H_q, d_h]
K, V                        [S, H_kv, d_h]
MLP intermediate            [S, D_ff]
```

An **attention head** is one learned subspace in which positions compare and
exchange information. Each head receives three learned projections of the
current token representations: a **query (Q)** describes what a position seeks,
a **key (K)** describes how a context position can match, and a **value (V)**
contains the information that matching position contributes. **Attention**
scores query-key matches, normalizes those scores into weights, and uses the
weights to mix values.

Linear layers can use matrix-matrix multiplication (**GEMM**, general matrix-
matrix multiply) because many token rows are available together. This creates
reuse and parallel work. A **causal mask** prevents a prompt position from
attending to later positions, which would reveal tokens that are still in its
future.

Prefill produces two outputs:

1. The final hidden state used to choose the first generated token.
2. A K and V entry for every prompt position at every layer, stored in the
   **KV cache**. Decode reads these saved entries instead of recomputing prior
   positions at every generation step.

### Decode

Decode processes one newly selected token, then repeats serially:

```text
new token ID                [1]
embedding                   [D]
Q                           [H_q, d_h]
new K, V                    [H_kv, d_h]
cached K, V                 [T, H_kv, d_h]
```

The large projections are now matrix-vector operations (**GEMV**, general
matrix-vector multiply). The current query must read keys and values from prior
positions. Weights are reused across generated tokens, but if they do not
remain in a sufficiently fast hardware cache, substantial weight data is read
again for each token. The KV cache grows with context and is also read
repeatedly.

That is why "inference speed" is not one property. Prefill and decode have
different shapes, reuse, bottlenecks, latency measures, and useful kernels.

## One decoder layer

A common pre-normalized decoder layer follows this dataflow. Exact ordering,
biases, normalization conventions, head counts, and activation details are
model-specific and must be read from the chosen architecture.

```text
residual input x
    |
    +--> RMSNorm --> Q/K/V projections --> RoPE --> attention
    |                                             --> output projection
    |                                                       |
    +-------------------------------------------------------+  residual add
                                                            |
                                                            v
                                                       residual y
                                                            |
    +--> RMSNorm --> gate/up projections --> SwiGLU --> down projection
    |                                                               |
    +---------------------------------------------------------------+  residual add
                                                                    |
                                                                    v
                                                              layer output
```

This is repeated $L$ times. The **residual stream** is the main token
representation that flows through the model. Each attention or feed-forward
branch computes a change and adds it back to that stream, preserving earlier
information while each layer refines it.

## Step-by-step decode trace

### 1. Token embedding

An **embedding table** stores one learned vector for every vocabulary item. The
latest token ID selects one row from its shape `[V, D]`, yielding the initial
residual vector $\mathbf{x} \in \mathbb{R}^D$.

### 2. Attention normalization

RMSNorm, or root-mean-square normalization, computes a scale from the $D$
features and uses it to keep the vector's magnitude controlled. This contains a
reduction followed by an elementwise pass.

### 3. Q/K/V projection

The normalized vector is multiplied by learned weight matrices to produce Q,
K, and V. The query represents what the current position is looking for; each
key represents how a position can be matched; and each value holds the content
retrieved when that position receives weight. These names describe roles in
the attention calculation, not literal text questions or database records.
During one-token decode the projections are GEMV-shaped operations. Their
outputs are viewed as heads rather than one flat feature axis.

### 4. Position encoding with RoPE

RoPE, or rotary position embeddings, rotates feature pairs in the new query and
key according to the current token position. The resulting query-key dot
products carry relative-position information. The value is not rotated.

### 5. KV-cache append

The new key and value are written to this layer's cache. Prior K/V entries are
not recomputed. Cache layout determines whether later attention reads are
contiguous and whether appends require copying or reallocation.

### 6. Scaled dot-product attention

Each query head computes dot products against allowed cached keys and divides
the scores by the square root of the head width. This complete operation is
called **scaled dot-product attention**. A **stable softmax** subtracts the
largest score before exponentiation, avoiding numerical overflow, and
normalizes the scores into nonnegative weights that sum to one. Those weights
combine the cached values.

With **multi-head attention (MHA)**, each query head has distinct K/V heads.
**Multi-query attention (MQA)** shares one K/V head across query heads.
**Grouped-query attention (GQA)** shares each K/V head across a group of query
heads. Sharing lowers KV storage and read traffic, but it is an architectural
choice established by the model's trained weights, not a runtime switch that
preserves an arbitrary model exactly.

A basic implementation materializes scores. An online implementation updates
the softmax statistics and weighted value as keys arrive. A tiled, Flash-style
implementation keeps score tiles and partial reductions in fast memory to avoid
writing the full score matrix. These approaches target the same mathematical
result with different memory behavior.

### 7. Attention output and residual

Head outputs are joined, projected back to width $D$, and added to the incoming
residual stream.

### 8. MLP normalization and SwiGLU

A second normalization feeds gate and up projections. SiLU is applied to the
gate, multiplied elementwise by the up projection, then projected from
$D_{ff}$ back to $D$. The result is added to the residual stream.

### 9. Repeat layers

Steps 2 through 8 run for every decoder layer. Each layer owns learned weights
and its own section of the KV cache.

### 10. Final normalization and logits

After the last layer, a final normalization is followed by an output projection
to `[V]` **logits**: unnormalized scores, one for every vocabulary item. Some
architectures tie this projection to the embedding table; others do not.

### 11. Sampling

The runtime transforms or filters logits according to the selected sampling
policy, such as greedy selection, temperature, top-k, or top-p, and chooses the
next token ID. Sampling changes selection behavior; it does not alter the
transformer computation that produced the logits.

### 12. Detokenization and repeat

The chosen token is converted back to bytes or text as permitted by the
tokenizer. Its ID becomes the input to the next decode step, and the KV cache now
has one more position.

## The two dominant memory budgets

### Weights

Ignoring format metadata, a rough lower bound is

$$
\text{weight bytes} \approx \text{parameter count} \times
\frac{\text{bits per stored weight}}{8}.
$$

Quantized formats also store scales and sometimes zero points. Runtime buffers,
alignment, mappings, and dequantized copies can increase resident memory.

### KV cache

For one sequence, a basic cache budget is

$$
\text{KV bytes} = 2 L T H_{kv} d_h b.
$$

The factor 2 is for K and V. This formula makes several tradeoffs concrete:
longer context increases memory linearly; MQA/GQA reduce $H_{kv}$; lower-precision
cache storage reduces $b$; sliding windows bound effective $T$.

## Questions to carry through the course

For every operation, ask:

1. What are the exact input and output shapes in prefill and decode?
2. Which values are reused, and where can they remain resident?
3. What is the arithmetic intensity: how many useful floating-point operations
   (FLOPs) are performed per byte moved?
4. Which intermediates are materialized only because operators are separate?
5. Which reduction order and dtype define acceptable numerical behavior?
6. Is a proposed optimization reducing computation, memory traffic, allocation,
   launch overhead, or synchronization?
7. Does the measurement isolate that cost?

The rest of Inference School turns each question into an executable problem.
