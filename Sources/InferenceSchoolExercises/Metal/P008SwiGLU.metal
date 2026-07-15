#include <metal_stdlib>
using namespace metal;

kernel void swiglu_gate(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        // TODO: Fuse SiLU(gate) with multiplication by up.
        output[index] = up[index] + gate[index] * 0.0f;
    }
}