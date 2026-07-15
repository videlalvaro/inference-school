#include <metal_stdlib>
using namespace metal;

kernel void quantized_cached_decode_attention(
  device const float *query [[buffer(0)]],
  device const char *keys [[buffer(1)]],
  device const char *values [[buffer(2)]],
  device const float *keyScales [[buffer(3)]],
  device const float *valueScales [[buffer(4)]],
  device float *output [[buffer(5)]],
  constant uint4 &shape [[buffer(6)]],
  constant uint4 &dimensions [[buffer(7)]],
  uint2 index [[thread_position_in_grid]]) {
  if (index.x >= dimensions.y || index.y >= shape.w) return;
  output[index.y * dimensions.y + index.x] = 0.0f;
}
