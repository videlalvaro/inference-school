#include <metal_stdlib>
using namespace metal;

constant uint q4ThreadgroupWidth = 256;
constant uint signedTwosComplementLowNibbleFirst = 0;

kernel void fused_q4_gemv(
  device const uchar *packedWeights [[buffer(0)]],
  device const float *scales [[buffer(1)]],
  device const float *input [[buffer(2)]],
  device float *output [[buffer(3)]],
  constant uint4 &shape [[buffer(4)]],
  uint lane [[thread_index_in_threadgroup]],
  uint row [[threadgroup_position_in_grid]]) {
  threadgroup float partials[q4ThreadgroupWidth];
  float sum = 0.0f;
  uint groupsPerRow = (shape.y + shape.z - 1) / shape.z;
  if (shape.w == signedTwosComplementLowNibbleFirst) {
    for (uint column = lane; column < shape.y; column += q4ThreadgroupWidth) {
      uint logicalIndex = row * shape.y + column;
      uchar byte = packedWeights[logicalIndex / 2];
      uchar nibble = (logicalIndex & 1) == 0 ? byte & 0x0f : byte >> 4;
      int quantized = nibble >= 8 ? int(nibble) - 16 : int(nibble);
      float scale = scales[row * groupsPerRow + column / shape.z];
      sum += float(quantized) * scale * input[column];
    }
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = q4ThreadgroupWidth / 2; stride > 0; stride >>= 1) {
    if (lane < stride) partials[lane] += partials[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) output[row] = partials[0];
}