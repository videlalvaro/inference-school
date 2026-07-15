#include <metal_stdlib>
using namespace metal;

kernel void vector_dot_partial(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device float *partialSums [[buffer(2)]],
    constant uint &elementCount [[buffer(3)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint groupIndex [[threadgroup_position_in_grid]]
) {
    if (localIndex == 0) {
        // TODO: Replace this placeholder with a parallel threadgroup reduction.
        partialSums[groupIndex] = 0.0f;
    }
}