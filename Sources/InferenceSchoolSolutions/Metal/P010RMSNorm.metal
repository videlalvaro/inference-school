#include <metal_stdlib>
using namespace metal;

constant uint threadgroupWidth = 256;

kernel void rmsnorm_rows(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[threadgroupWidth];
    float localSumSquares = 0.0f;
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        float value = input[row * shape.y + column];
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
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        output[row * shape.y + column] = input[row * shape.y + column] * inverseRMS * gamma[column];
    }
}