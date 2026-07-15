public struct ProblemSummary: Sendable {
    public let id: String
    public let title: String
    public let concept: String
    public let engineUse: String
    public let chapterPath: String
    public let isAvailable: Bool

    public init(
        id: String,
        title: String,
        concept: String,
        engineUse: String,
        chapterPath: String,
        isAvailable: Bool
    ) {
        self.id = id
        self.title = title
        self.concept = concept
        self.engineUse = engineUse
        self.chapterPath = chapterPath
        self.isAvailable = isAvailable
    }
}

public enum Course {
    public static let title = "Inference School"

    public static let availableProblems = [
        ProblemSummary(
            id: "001",
            title: "Vector Dot Product",
            concept: "reductions, floating-point accumulation, Metal threadgroups",
            engineUse: "one output element in GEMV and one query-key attention score",
            chapterPath: "Problems/001-vector-dot/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "002",
            title: "Tensor Storage and Strides",
            concept: "row-major storage, shapes, strides, checked views",
            engineUse: "the address calculation and layout contract for every later operator",
            chapterPath: "Problems/002-tensor-storage-and-strides/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "003",
            title: "Transpose and Tiled Copy",
            concept: "layout conversion, coalescing, threadgroup tiles",
            engineUse: "materializing layouts that make later matrix and attention kernels efficient",
            chapterPath: "Problems/003-transpose-and-tiled-copy/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "004",
            title: "Matrix-Vector Multiplication",
            concept: "GEMV, row reductions, bandwidth-bound projection",
            engineUse: "one-token decode projections for attention and the MLP",
            chapterPath: "Problems/004-matrix-vector-multiplication/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "005",
            title: "Matrix-Matrix Multiplication",
            concept: "GEMM, data reuse, tiled matrix multiplication",
            engineUse: "multi-token prefill projections and batched linear layers",
            chapterPath: "Problems/005-matrix-matrix-multiplication/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "006",
            title: "Build a Roofline Measurement",
            concept: "arithmetic intensity, machine ceilings, measured throughput",
            engineUse: "deciding whether an inference operator needs less traffic, more reuse, or less overhead",
            chapterPath: "Problems/006-roofline-measurement/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "007",
            title: "ReLU, GELU, and SiLU",
            concept: "elementwise nonlinearities, GELU conventions, memory passes",
            engineUse: "nonlinear transformations inside transformer feed-forward blocks",
            chapterPath: "Problems/007-activations/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "008",
            title: "SwiGLU Feed-Forward Gate",
            concept: "gate/up/down projections, SiLU gating, elementwise fusion",
            engineUse: "the gated MLP sublayer in a modern decoder block",
            chapterPath: "Problems/008-swiglu-feed-forward/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "009",
            title: "Numerically Stable Softmax",
            concept: "max subtraction, exponential normalization, row reductions",
            engineUse: "turning masked attention scores into probabilities",
            chapterPath: "Problems/009-stable-softmax/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "010",
            title: "RMSNorm",
            concept: "root-mean-square normalization, epsilon, reduction precision",
            engineUse: "pre-normalizing residual streams before attention and MLP projections",
            chapterPath: "Problems/010-rmsnorm/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "011",
            title: "Residual Streams and Precision",
            concept: "residual accumulation, Float16 round trips, downcast placement",
            engineUse: "preserving the shared residual state across many decoder layers",
            chapterPath: "Problems/011-residual-stream-precision/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "012",
            title: "Fuse Norm, Scale, and Projection Input",
            concept: "operator fusion, eliminated intermediates, fused reduction and GEMV",
            engineUse: "feeding normalized residual values directly into decoder projections",
            chapterPath: "Problems/012-fused-rmsnorm-gemv/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "013",
            title: "Embedding Lookup and Tied Output Weights",
            concept: "indexed gather, vocabulary tables, tied unembedding projection",
            engineUse: "mapping token IDs into the residual stream and final hidden states back to vocabulary logits",
            chapterPath: "Problems/013-embedding-lookup/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "014",
            title: "Q/K/V Projections and Head Views",
            concept: "explicit projection shapes, query heads, KV heads, contiguous head views",
            engineUse: "building the query, key, and value tensors consumed by attention",
            chapterPath: "Problems/014-qkv-projections/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "015",
            title: "Rotary Position Embeddings",
            concept: "adjacent-pair rotations, frequencies, offsets, partial rotary dimensions",
            engineUse: "encoding absolute token positions into queries and keys before score computation",
            chapterPath: "Problems/015-rope/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "016",
            title: "Causal Attention for One Head",
            concept: "scaled query-key scores, causal masking, stable softmax, weighted values",
            engineUse: "the readable materialized oracle for decoder self-attention",
            chapterPath: "Problems/016-causal-attention/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "017",
            title: "Multi-Head Attention",
            concept: "independent head attention, head-parallel execution, concatenated output layout",
            engineUse: "running several learned attention subspaces in one decoder layer",
            chapterPath: "Problems/017-multi-head-attention/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "018",
            title: "MHA, MQA, and GQA",
            concept: "query-to-KV head grouping, shared KV state, cache memory formulas",
            engineUse: "matching model-specific attention architecture while reducing KV-cache traffic",
            chapterPath: "Problems/018-mha-mqa-gqa/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "019",
            title: "Online Softmax Attention",
            concept: "streaming max, normalizer, and weighted-output recurrence",
            engineUse: "computing stable attention without materializing score rows",
            chapterPath: "Problems/019-online-attention/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "020",
            title: "Tiled Fused Attention",
            concept: "key tiling, on-chip score tiles, fused online normalization",
            engineUse: "removing the quadratic score-matrix traffic in attention prefill",
            chapterPath: "Problems/020-tiled-fused-attention/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "021",
            title: "Sliding-Window Attention",
            concept: "inclusive causal windows, absolute positions, bounded attention work",
            engineUse: "limiting cache reads and attention computation to recent context",
            chapterPath: "Problems/021-sliding-window-attention/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "022",
            title: "Preallocate and Append K/V",
            concept: "fixed-capacity storage, logical positions, layer isolation",
            engineUse: "retaining projected keys and values without allocation in the decode loop",
            chapterPath: "Problems/022-preallocate-and-append-kv/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "023",
            title: "Cached Single-Token Attention",
            concept: "append-then-attend decode, cache reuse, absolute positions",
            engineUse: "computing one decoder attention output without reprojecting prior tokens",
            chapterPath: "Problems/023-cached-single-token-attention/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "024",
            title: "KV Layout Shootout",
            concept: "token-major and head-major offsets, access traces, shape-specific benchmarks",
            engineUse: "choosing a cache layout from the decoder's measured read pattern",
            chapterPath: "Problems/024-kv-layout-shootout/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "025",
            title: "Shared KV Heads",
            concept: "GQA cache mapping, division groups, MHA MQA GQA byte budgets",
            engineUse: "allocating and reading cache state using the checkpoint's KV-head count",
            chapterPath: "Problems/025-shared-kv-heads/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "026",
            title: "Ring-Buffer Sliding Cache",
            concept: "physical wraparound, monotonic positions, chronological reads",
            engineUse: "bounding live cache memory for layers with a fixed attention window",
            chapterPath: "Problems/026-ring-buffer-sliding-cache/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "027",
            title: "Paged KV Allocation",
            concept: "page tables, allocation, free and reuse, fragmentation accounting",
            engineUse: "growing and reclaiming sequence cache state without physical contiguity",
            chapterPath: "Problems/027-paged-kv-allocation/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "028",
            title: "Quantized KV Cache",
            concept: "symmetric INT8 vectors, scale metadata, on-read dequantization",
            engineUse: "reducing decoder cache bytes and bandwidth with measured output error",
            chapterPath: "Problems/028-quantized-kv-cache/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "029",
            title: "Symmetric INT8 Quantization",
            concept: "signed ranges, deterministic rounding, scales, reconstruction error",
            engineUse: "converting Float32 model weights into an explicit compact representation",
            chapterPath: "Problems/029-symmetric-int8-quantization/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "030",
            title: "Per-Channel and Groupwise Scales",
            concept: "output-channel groups, tail metadata, error versus byte tradeoffs",
            engineUse: "preserving local weight ranges while bounding scale metadata",
            chapterPath: "Problems/030-per-channel-and-groupwise-scales/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "031",
            title: "Pack and Unpack INT4",
            concept: "signed nibbles, two's-complement, byte order, odd-count padding",
            engineUse: "the byte-exact Q4 checkpoint and kernel interchange format",
            chapterPath: "Problems/031-pack-unpack-int4/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "032",
            title: "Dequantize Then GEMV",
            concept: "Q4 materialization, Float32 temporary traffic, staged baseline",
            engineUse: "a readable parity baseline for quantized one-token projections",
            chapterPath: "Problems/032-dequantize-then-gemv/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "033",
            title: "Fused Q4 GEMV",
            concept: "inline unpacking, group scales, fused dequantization, Metal reduction",
            engineUse: "streaming compact decoder weights directly into one-token projections",
            chapterPath: "Problems/033-fused-q4-gemv/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "034",
            title: "Quantization Error Propagation",
            concept: "layer captures, cosine similarity, RMSE, argmax, convention diagnosis",
            engineUse: "qualifying a converted model before attributing output differences to precision",
            chapterPath: "Problems/034-quantization-error-propagation/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "035",
            title: "One Decoder Transformer Block",
            concept: "pre-norm ordering, RoPE, causal GQA, residuals, SwiGLU",
            engineUse: "the reusable layer computation shared by prompt prefill and token decode",
            chapterPath: "Problems/035-decoder-transformer-block/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "036",
            title: "Parse a Model Weight Format",
            concept: "little-endian binary parsing, JSON metadata, bounds and shape validation",
            engineUse: "loading named checkpoint tensors into the decoder block without trusting offsets",
            chapterPath: "Problems/036-model-weight-format/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "037",
            title: "Tokenization and Detokenization",
            concept: "UTF-8 bytes, ranked BPE merges, special tokens, lossless byte decode",
            engineUse: "mapping prompt text to checkpoint-compatible token IDs and generated IDs back to bytes",
            chapterPath: "Problems/037-byte-bpe-tokenization/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "038",
            title: "Logits and Sampling",
            concept: "greedy selection, temperature, top-k, top-p, deterministic random draws",
            engineUse: "selecting the next token from vocabulary logits in the Problem 040 decode loop",
            chapterPath: "Problems/038-logits-and-sampling/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "039",
            title: "Prompt Prefill",
            concept: "multi-token embeddings, ordered decoder layers, cache population, final-position logits",
            engineUse: "processing a prompt with GEMM-shaped projections and seeding every layer's KV cache",
            chapterPath: "Problems/039-prompt-prefill/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "040",
            title: "Autoregressive Decode",
            concept: "one-token execution, append-before-attend cache state, deterministic generation",
            engineUse: "generating tokens without recomputing prior keys and values",
            chapterPath: "Problems/040-autoregressive-decode/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "041",
            title: "Buffer Reuse and Memory Planning",
            concept: "buffer lifetimes, alignment, arena allocation, deterministic first-fit and best-fit",
            engineUse: "planning reusable intermediate storage for prefill and one-token decode",
            chapterPath: "Problems/041-buffer-reuse-and-memory-planning/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "042",
            title: "Checkpoint Parity and First Divergence",
            concept: "named captures, serialized reference artifacts, numerical metrics, fault localization",
            engineUse: "finding the first model boundary that differs from a trusted capture set",
            chapterPath: "Problems/042-checkpoint-parity-and-first-divergence/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "043",
            title: "Fuse RMSNorm and Q/K/V Projections",
            concept: "subgraph fusion, eliminated intermediates, dispatch and transfer accounting",
            engineUse: "feeding one normalized residual stream directly into all attention projections",
            chapterPath: "Problems/043-fused-qkv/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "044",
            title: "Profile Prefill and Decode Separately",
            concept: "monotonic timing, warmup, latency distributions, stage-specific token rates",
            engineUse: "distinguishing prompt throughput from serial decode latency before optimizing either path",
            chapterPath: "Problems/044-prefill-decode-profiling/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "045",
            title: "Static and Continuous Batching",
            concept: "request arrivals, slot refill, discrete-event scheduling, latency-throughput tradeoffs",
            engineUse: "sharing prefill and decode work across independent generation requests",
            chapterPath: "Problems/045-batch-scheduling/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "046",
            title: "Speculative Decoding",
            concept: "draft proposals, target verification, rejection correction, expected-cost modeling",
            engineUse: "emitting several target-distributed tokens per serial target verification step",
            chapterPath: "Problems/046-speculative-decoding/README.md",
            isAvailable: true
        ),
        ProblemSummary(
            id: "047",
            title: "Capstone Inference Engine",
            concept: "compatible tokenization, prefill, cached decode, measurement, parity, engineering judgment",
            engineUse: "running and defending one complete educational text-to-token inference path",
            chapterPath: "Problems/047-capstone-inference-engine/README.md",
            isAvailable: true
        ),
    ]

    public static func problem(id: String) -> ProblemSummary? {
        availableProblems.first { $0.id == id }
    }
}