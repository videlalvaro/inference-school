import Foundation
import InferenceSchoolCore

public enum P007ActivationSolution {
    public static func apply(
        _ input: FloatTensor,
        activation: Activation
    ) throws -> FloatTensor {
        let output = input.storage.map { value in
            switch activation {
            case .relu:
                return max(0, value)
            case .geluTanhApproximation:
                let cubic = value * value * value
                let inner = sqrt(2 / Float.pi) * (value + 0.044715 * cubic)
                return 0.5 * value * (1 + tanh(inner))
            case .silu:
                return value / (1 + exp(-value))
            }
        }
        return try FloatTensor(output, shape: input.shape)
    }
}