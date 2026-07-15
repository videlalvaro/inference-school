#include <metal_stdlib>
using namespace metal;

kernel void softmax_rows(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    // TODO: Reduce the row maximum, then the shifted exponential sum.
    for (uint column = localIndex; column < shape.y; column += 256) {
        output[row * shape.y + column] = input[row * shape.y + column] * 0.0f;
    }
}