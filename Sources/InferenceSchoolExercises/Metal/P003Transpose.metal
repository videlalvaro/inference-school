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
    uint inputRow = groupPosition.y * tileWidth + localPosition.y;
    uint inputColumn = groupPosition.x * tileWidth + localPosition.x;
    if (inputRow < shape.x && inputColumn < shape.y) {
        // TODO: Stage a tile in threadgroup memory and store it transposed.
        output[inputColumn * shape.x + inputRow] = 0.0f;
    }
}