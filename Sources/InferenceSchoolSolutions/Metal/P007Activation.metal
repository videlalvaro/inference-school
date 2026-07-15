#include <metal_stdlib>
using namespace metal;

kernel void activation_elementwise(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &activation [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }
    float value = input[index];
    if (activation == 0) {
        output[index] = max(0.0f, value);
    } else if (activation == 1) {
        float cubic = value * value * value;
        output[index] = 0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * cubic)));
    } else {
        output[index] = value / (1.0f + exp(-value));
    }
}