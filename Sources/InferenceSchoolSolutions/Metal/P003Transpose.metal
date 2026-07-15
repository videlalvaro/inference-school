#include <metal_stdlib>
using namespace metal;

constant uint tileWidth = 16;

kernel void tiled_transpose(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],
    uint2 localPosition [[thread_position_in_threadgroup]],
    uint2 groupPosition [[threadgroup_position_in_grid]]
) {
    threadgroup float tile[tileWidth][tileWidth + 1];

    uint inputRow = groupPosition.y * tileWidth + localPosition.y;
    uint inputColumn = groupPosition.x * tileWidth + localPosition.x;
    if (inputRow < shape.x && inputColumn < shape.y) {
        tile[localPosition.y][localPosition.x] = input[inputRow * shape.y + inputColumn];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint outputRow = groupPosition.x * tileWidth + localPosition.y;
    uint outputColumn = groupPosition.y * tileWidth + localPosition.x;
    if (outputRow < shape.y && outputColumn < shape.x) {
        output[outputRow * shape.x + outputColumn] = tile[localPosition.x][localPosition.y];
    }
}