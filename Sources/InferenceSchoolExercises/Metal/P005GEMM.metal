#include <metal_stdlib>
using namespace metal;

constant uint tileWidth = 16;

kernel void tiled_gemm(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint4 &shape [[buffer(3)]],
    uint2 localPosition [[thread_position_in_threadgroup]],
    uint2 groupPosition [[threadgroup_position_in_grid]]
) {
    uint row = groupPosition.y * tileWidth + localPosition.y;
    uint column = groupPosition.x * tileWidth + localPosition.x;
    if (row < shape.x && column < shape.z) {
        // TODO: Cooperatively load A/B tiles and accumulate this output cell.
        output[row * shape.z + column] = 0.0f;
    }
}