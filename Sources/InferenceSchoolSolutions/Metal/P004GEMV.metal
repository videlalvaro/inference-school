#include <metal_stdlib>
using namespace metal;

constant uint threadgroupWidth = 256;

kernel void gemv_rows(
    device const float *matrix [[buffer(0)]],
    device const float *vector [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partials[threadgroupWidth];
    float sum = 0.0f;
    for (uint column = localIndex; column < shape.y; column += threadgroupWidth) {
        sum += matrix[row * shape.y + column] * vector[column];
    }
    partials[localIndex] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroupWidth / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            partials[localIndex] += partials[localIndex + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (localIndex == 0) {
        output[row] = partials[0];
    }
}