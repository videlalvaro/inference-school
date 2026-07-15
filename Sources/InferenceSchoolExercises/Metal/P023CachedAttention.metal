#include <metal_stdlib>
using namespace metal;

kernel void cached_decode_attention(
  device const float *query [[buffer(0)]],
  device const float *keys [[buffer(1)]],
  device const float *values [[buffer(2)]],
  device float *output [[buffer(3)]],
  constant uint4 &shape [[buffer(4)]],
  constant uint4 &dimensions [[buffer(5)]],
  uint2 index [[thread_position_in_grid]]) {
  if (index.x >= dimensions.y || index.y >= shape.w) return;
  output[index.y * dimensions.y + index.x] = 0.0f;
}
