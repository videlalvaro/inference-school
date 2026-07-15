#include <metal_stdlib>
using namespace metal;

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
    // TODO: Reduce RMS and the projection in this dispatch without an intermediate tensor.
    if (localIndex == 0) {
        output[outputRow] = input[0] * gamma[0] * weights[outputRow * shape.y] * epsilon * 0.0f;
    }
}