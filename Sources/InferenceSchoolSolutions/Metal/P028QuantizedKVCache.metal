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
  uint feature = index.x;
  uint queryHead = index.y;
  if (feature >= dimensions.y || queryHead >= shape.w) return;

  uint kvHead = queryHead / dimensions.z;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float weightedValue = 0.0f;
  for (uint token = 0; token < shape.z; ++token) {
    uint vectorIndex = ((shape.x * shape.y + token) * dimensions.x + kvHead);
    uint elementBase = vectorIndex * dimensions.y;
    float keyScale = keyScales[vectorIndex];
    float valueScale = valueScales[vectorIndex];
    float score = 0.0f;
    for (uint d = 0; d < dimensions.y; ++d) {
      score += query[queryHead * dimensions.y + d]
        * float(keys[elementBase + d]) * keyScale;
    }
    score *= rsqrt(float(dimensions.y));
    float nextMaximum = max(maximum, score);
    float alpha = isfinite(maximum) ? exp(maximum - nextMaximum) : 0.0f;
    float beta = exp(score - nextMaximum);
    denominator = denominator * alpha + beta;
    weightedValue = weightedValue * alpha
      + beta * float(values[elementBase + feature]) * valueScale;
    maximum = nextMaximum;
  }
  output[queryHead * dimensions.y + feature] = weightedValue / denominator;
}
