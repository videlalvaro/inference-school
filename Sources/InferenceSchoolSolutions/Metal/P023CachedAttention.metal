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
  uint feature = index.x;
  uint queryHead = index.y;
  if (feature >= dimensions.y || queryHead >= shape.w) return;

  uint kvHead = queryHead / dimensions.z;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float weightedValue = 0.0f;
  for (uint token = 0; token < shape.z; ++token) {
    uint tokenBase = (((shape.x * shape.y) + token) * dimensions.x + kvHead) * dimensions.y;
    float score = 0.0f;
    for (uint d = 0; d < dimensions.y; ++d) {
      score += query[queryHead * dimensions.y + d] * keys[tokenBase + d];
    }
    score *= rsqrt(float(dimensions.y));
    float nextMaximum = max(maximum, score);
    float alpha = isfinite(maximum) ? exp(maximum - nextMaximum) : 0.0f;
    float beta = exp(score - nextMaximum);
    denominator = denominator * alpha + beta;
    weightedValue = weightedValue * alpha + beta * values[tokenBase + feature];
    maximum = nextMaximum;
  }
  output[queryHead * dimensions.y + feature] = weightedValue / denominator;
}
