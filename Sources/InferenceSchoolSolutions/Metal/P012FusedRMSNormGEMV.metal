#include <metal_stdlib>
using namespace metal;

constant uint threadgroupWidth = 256;

kernel void fused_rmsnorm_gemv(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint2 &shape [[buffer(4)]],
    constant float &epsilon [[buffer(5)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint outputRow [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[threadgroupWidth];
    float localSumSquares = 0.0f;
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        float value = input[column];
        localSumSquares += value * value;
    }
    scratch[localIndex] = localSumSquares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroupWidth / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            scratch[localIndex] += scratch[localIndex + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverseRMS = rsqrt(scratch[0] / float(shape.y) + epsilon);
    float localProjection = 0.0f;
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        float normalized = input[column] * inverseRMS * gamma[column];
        localProjection += weights[outputRow * shape.y + column] * normalized;
    }
    scratch[localIndex] = localProjection;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroupWidth / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            scratch[localIndex] += scratch[localIndex + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (localIndex == 0) {
        output[outputRow] = scratch[0];
    }
}