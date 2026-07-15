#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_rows(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    // TODO: Reduce mean square, add epsilon, then scale by gamma.
    for (uint column = localIndex; column < shape.y; column += 256) {
        output[row * shape.y + column] = input[row * shape.y + column] * gamma[column] * epsilon * 0.0f;
    }
}