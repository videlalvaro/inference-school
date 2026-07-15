import Foundation
import InferenceSchoolCore

public enum P012FusedRMSNormGEMVSolution {
    public static func baseline(_ input: FloatTensor, gamma: FloatTensor, weights: FloatTensor, epsilon: Float) throws -> FloatTensor {
        try validate(input, gamma: gamma, weights: weights, epsilon: epsilon)
        let width = input.shape[0]
        var sumSquares: Float = 0
        for value in input.storage { sumSquares += value * value }
        let inverseRMS = 1 / sqrt(sumSquares / Float(width) + epsilon)
        let normalized = (0..<width).map { input.storage[$0] * inverseRMS * gamma.storage[$0] }
        var output = Array(repeating: Float.zero, count: weights.shape[0])
        for row in output.indices {
            for column in 0..<width { output[row] += weights.storage[row * width + column] * normalized[column] }
        }
        return try FloatTensor(output, shape: [output.count])
    }

    private static func validate(_ input: FloatTensor, gamma: FloatTensor, weights: FloatTensor, epsilon: Float) throws {
        guard input.rank == 1 else { throw TensorError.rankMismatch(expected: 1, actual: input.rank) }
        guard gamma.rank == 1 else { throw TensorError.rankMismatch(expected: 1, actual: gamma.rank) }
        guard weights.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: weights.rank) }
        guard !input.storage.isEmpty else { throw FusedRMSNormGEMVError.emptyInput }
        guard weights.shape[1] == input.shape[0] else { throw FusedRMSNormGEMVError.inputWidthMismatch(expected: weights.shape[1], actual: input.shape[0]) }
        guard gamma.shape[0] == input.shape[0] else { throw FusedRMSNormGEMVError.gammaWidthMismatch(expected: input.shape[0], actual: gamma.shape[0]) }
        guard epsilon.isFinite, epsilon > 0 else { throw FusedRMSNormGEMVError.invalidEpsilon(epsilon) }
    }
}