import Foundation
import InferenceSchoolCore

public enum P008SwiGLUSolution {
    public static func apply(_ input: FloatTensor, gateWeights: FloatTensor, upWeights: FloatTensor, downWeights: FloatTensor) throws -> FloatTensor {
        try validate(input, gateWeights: gateWeights, upWeights: upWeights, downWeights: downWeights)
        let hidden = gateWeights.shape[0]
        let width = input.shape[0]
        var gated = Array(repeating: Float.zero, count: hidden)
        for row in 0..<hidden {
            var gate: Float = 0
            var up: Float = 0
            for column in 0..<width {
                gate += gateWeights.storage[row * width + column] * input.storage[column]
                up += upWeights.storage[row * width + column] * input.storage[column]
            }
            gated[row] = (gate / (1 + exp(-gate))) * up
        }
        var output = Array(repeating: Float.zero, count: downWeights.shape[0])
        for row in output.indices {
            for column in 0..<hidden {
                output[row] += downWeights.storage[row * hidden + column] * gated[column]
            }
        }
        return try FloatTensor(output, shape: [output.count])
    }

    public static func gate(_ gate: FloatTensor, up: FloatTensor) throws -> FloatTensor {
        guard gate.shape == up.shape else { throw SwiGLUError.gateValueShapeMismatch(gate: gate.shape, up: up.shape) }
        let output = zip(gate.storage, up.storage).map { gateValue, upValue in
            (gateValue / (1 + exp(-gateValue))) * upValue
        }
        return try FloatTensor(output, shape: gate.shape)
    }

    private static func validate(_ input: FloatTensor, gateWeights: FloatTensor, upWeights: FloatTensor, downWeights: FloatTensor) throws {
        guard input.rank == 1 else { throw TensorError.rankMismatch(expected: 1, actual: input.rank) }
        guard gateWeights.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: gateWeights.rank) }
        guard upWeights.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: upWeights.rank) }
        guard downWeights.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: downWeights.rank) }
        guard gateWeights.shape == upWeights.shape else { throw SwiGLUError.hiddenProjectionShapeMismatch(gate: gateWeights.shape, up: upWeights.shape) }
        guard gateWeights.shape[1] == input.shape[0] else { throw SwiGLUError.inputWidthMismatch(expected: gateWeights.shape[1], actual: input.shape[0]) }
        guard downWeights.shape[1] == gateWeights.shape[0] else { throw SwiGLUError.downWidthMismatch(expected: gateWeights.shape[0], actual: downWeights.shape[1]) }
    }
}