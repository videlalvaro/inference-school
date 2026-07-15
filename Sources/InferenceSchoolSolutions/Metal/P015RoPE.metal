#include <metal_stdlib>
using namespace metal;

kernel void rope_adjacent_pairs(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint4 &shape [[buffer(2)]],
    constant uint &positionOffset [[buffer(3)]],
    constant float &ropeBase [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint count = shape.x * shape.y * shape.w;
    if (index >= count) {
        return;
    }
    uint pair = index % shape.w;
    uint head = (index / shape.w) % shape.y;
    uint sequence = index / (shape.w * shape.y);
    uint feature = pair * 2;
    uint tensorIndex = (sequence * shape.y + head) * shape.z + feature;
    float angle = float(positionOffset + sequence)
        / pow(ropeBase, float(feature) / float(shape.w * 2));
    float cosine = cos(angle);
    float sine = sin(angle);
    float first = input[tensorIndex];
    float second = input[tensorIndex + 1];
    output[tensorIndex] = first * cosine - second * sine;
    output[tensorIndex + 1] = first * sine + second * cosine;
}