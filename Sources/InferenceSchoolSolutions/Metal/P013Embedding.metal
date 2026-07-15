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
    uint sequence = index / shape.y;
    uint feature = index % shape.y;
    embeddings[index] = table[tokenIDs[sequence] * shape.y + feature];
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
    uint sequence = index / shape.x;
    uint vocabulary = index % shape.x;
    float sum = 0.0f;
    for (uint feature = 0; feature < shape.y; ++feature) {
        sum += embeddings[sequence * shape.y + feature]
            * table[vocabulary * shape.y + feature];
    }
    logits[index] = sum;
}