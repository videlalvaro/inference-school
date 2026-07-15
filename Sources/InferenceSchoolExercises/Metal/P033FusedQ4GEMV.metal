#include <metal_stdlib>
using namespace metal;

kernel void fused_q4_gemv(
  device const uchar *packedWeights [[buffer(0)]],
  device const float *scales [[buffer(1)]],
  device const float *input [[buffer(2)]],
  device float *output [[buffer(3)]],
  constant uint4 &shape [[buffer(4)]],
  uint lane [[thread_index_in_threadgroup]],
  uint row [[threadgroup_position_in_grid]]) {
  if (lane == 0 && row < shape.x) output[row] = 0.0f;
}