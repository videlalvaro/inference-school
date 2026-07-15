#include <metal_stdlib>
using namespace metal;

kernel void gemv_rows(
    device const float *matrix [[buffer(0)]],
    device const float *vector [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (localIndex == 0 && row < shape.x) {
        // TODO: Cooperatively reduce this row against the vector.
        output[row] = 0.0f;
    }
}