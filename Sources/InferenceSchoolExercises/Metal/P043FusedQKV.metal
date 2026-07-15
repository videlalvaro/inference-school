#include <metal_stdlib>
using namespace metal;

constant uint fusedThreadgroupWidth = 256;

kernel void fused_rmsnorm_qkv(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *queryWeights [[buffer(2)]],
    device const float *keyWeights [[buffer(3)]],
    device const float *valueWeights [[buffer(4)]],
    device float *queries [[buffer(5)]],
    device float *keys [[buffer(6)]],
    device float *values [[buffer(7)]],
    constant uint4 &shape [[buffer(8)]],
    constant float &epsilon [[buffer(9)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[fusedThreadgroupWidth];
    float localSumSquares = 0.0f;
    for (uint feature = localIndex; feature < shape.y; feature += fusedThreadgroupWidth) {
        float value = input[token * shape.y + feature];
        localSumSquares += value * value;
    }
    scratch[localIndex] = localSumSquares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = fusedThreadgroupWidth / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            scratch[localIndex] += scratch[localIndex + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverseRMS = rsqrt(scratch[0] / float(shape.y) + epsilon);

    // TODO: use the normalized row to evaluate Q, K, and V channel dot products.
    uint combinedWidth = shape.z + 2 * shape.w;
    for (uint channel = localIndex; channel < combinedWidth; channel += fusedThreadgroupWidth) {
        if (channel < shape.z) {
            queries[token * shape.z + channel] = 0.0f * inverseRMS;
        } else if (channel < shape.z + shape.w) {
            keys[token * shape.w + channel - shape.z] = 0.0f * inverseRMS;
        } else {
            values[token * shape.w + channel - shape.z - shape.w] = 0.0f * inverseRMS;
        }
    }
}