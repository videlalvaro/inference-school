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
    threadgroup float lhsTile[tileWidth][tileWidth];
    threadgroup float rhsTile[tileWidth][tileWidth];

    uint row = groupPosition.y * tileWidth + localPosition.y;
    uint column = groupPosition.x * tileWidth + localPosition.x;
    float sum = 0.0f;
    uint tileCount = (shape.y + tileWidth - 1) / tileWidth;

    for (uint tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        uint lhsColumn = tileIndex * tileWidth + localPosition.x;
        uint rhsRow = tileIndex * tileWidth + localPosition.y;
        lhsTile[localPosition.y][localPosition.x] = row < shape.x && lhsColumn < shape.y
            ? lhs[row * shape.y + lhsColumn]
            : 0.0f;
        rhsTile[localPosition.y][localPosition.x] = rhsRow < shape.y && column < shape.z
            ? rhs[rhsRow * shape.z + column]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < tileWidth; ++inner) {
            sum += lhsTile[localPosition.y][inner] * rhsTile[inner][localPosition.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < shape.x && column < shape.z) {
        output[row * shape.z + column] = sum;
    }
}