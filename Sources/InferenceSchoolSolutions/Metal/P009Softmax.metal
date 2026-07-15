#include <metal_stdlib>
using namespace metal;

constant uint threadgroupWidth = 256;

kernel void softmax_rows(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[threadgroupWidth];
    float localMaximum = -INFINITY;
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        localMaximum = max(localMaximum, input[row * shape.y + column]);
    }
    scratch[localIndex] = localMaximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroupWidth / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            scratch[localIndex] = max(scratch[localIndex], scratch[localIndex + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = scratch[0];
    float localSum = 0.0f;
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        localSum += exp(input[row * shape.y + column] - maximum);
    }
    scratch[localIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroupWidth / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            scratch[localIndex] += scratch[localIndex + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = scratch[0];
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        output[row * shape.y + column] = exp(input[row * shape.y + column] - maximum) / denominator;
    }
}