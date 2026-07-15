#include <metal_stdlib>
using namespace metal;

constant uint threadgroupWidth = 256;

kernel void vector_dot_partial(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device float *partialSums [[buffer(2)]],
    constant uint &elementCount [[buffer(3)]],
    uint globalIndex [[thread_position_in_grid]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint groupIndex [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[threadgroupWidth];
    scratch[localIndex] = globalIndex < elementCount
        ? lhs[globalIndex] * rhs[globalIndex]
        : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroupWidth / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            scratch[localIndex] += scratch[localIndex + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (localIndex == 0) {
        partialSums[groupIndex] = scratch[0];
    }
}