#include <metal_stdlib>
using namespace metal;

kernel void embedding_lookup(
    device const float *table [[buffer(0)]],
    device const uint *tokenIDs [[buffer(1)]],
    device float *embeddings [[buffer(2)]],
    constant uint4 &shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint count = shape.y * shape.z;
    if (index >= count) {
        return;
    }
    embeddings[index] = 0.0f;
}

kernel void tied_unembedding(
    device const float *embeddings [[buffer(0)]],
    device const float *table [[buffer(1)]],
    device float *logits [[buffer(2)]],
    constant uint4 &shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint count = shape.x * shape.z;
    if (index >= count) {
        return;
    }
    logits[index] = 0.0f;
}