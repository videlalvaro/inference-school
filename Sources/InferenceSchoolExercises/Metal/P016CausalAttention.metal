#include <metal_stdlib>
using namespace metal;

kernel void causal_attention_scores(device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]], device float *scores [[buffer(2)]], constant uint4 &shape [[buffer(3)]], constant uint2 &offsets [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index < shape.x * shape.y) { scores[index] = 0.0f; }
}

kernel void causal_attention_apply(device const float *scores [[buffer(0)]], device const float *values [[buffer(1)]], device float *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]], uint query [[thread_position_in_grid]]) {
    if (query >= shape.x) { return; }
    for (uint feature = 0; feature < shape.z; ++feature) { output[query * shape.z + feature] = 0.0f; }
}