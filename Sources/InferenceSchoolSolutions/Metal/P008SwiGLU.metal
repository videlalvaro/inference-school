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
        float gateValue = gate[index];
        output[index] = (gateValue / (1.0f + exp(-gateValue))) * up[index];
    }
}