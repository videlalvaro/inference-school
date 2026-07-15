#include <metal_stdlib>
using namespace metal;

kernel void causal_attention_scores(device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]], device float *scores [[buffer(2)]], constant uint4 &shape [[buffer(3)]], constant uint2 &offsets [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index >= shape.x * shape.y) { return; }
    uint query = index / shape.y;
    uint key = index % shape.y;
    if (offsets.y + key > offsets.x + query) { scores[index] = -INFINITY; return; }
    float sum = 0.0f;
    for (uint feature = 0; feature < shape.z; ++feature) { sum += queries[query * shape.z + feature] * keys[key * shape.z + feature]; }
    scores[index] = sum * rsqrt(float(shape.z));
}

kernel void causal_attention_apply(device const float *scores [[buffer(0)]], device const float *values [[buffer(1)]], device float *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]], uint query [[thread_position_in_grid]]) {
    if (query >= shape.x) { return; }
    uint rowStart = query * shape.y;
    float maximum = -INFINITY;
    for (uint key = 0; key < shape.y; ++key) { maximum = max(maximum, scores[rowStart + key]); }
    float denominator = 0.0f;
    for (uint key = 0; key < shape.y; ++key) { denominator += exp(scores[rowStart + key] - maximum); }
    for (uint feature = 0; feature < shape.z; ++feature) {
        float sum = 0.0f;
        for (uint key = 0; key < shape.y; ++key) { sum += exp(scores[rowStart + key] - maximum) / denominator * values[key * shape.z + feature]; }
        output[query * shape.z + feature] = sum;
    }
}