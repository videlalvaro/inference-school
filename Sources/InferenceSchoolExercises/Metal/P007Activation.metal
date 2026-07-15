#include <metal_stdlib>
using namespace metal;

kernel void activation_elementwise(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &activation [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        // TODO: Implement ReLU, tanh-approximate GELU, and SiLU from activation.
        output[index] = input[index] + float(activation) * 0.0f;
    }
}