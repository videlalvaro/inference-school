import Foundation
import InferenceSchoolCore

public enum P010RMSNormSolution {
    public static func apply(_ input: FloatTensor, gamma: FloatTensor, epsilon: Float) throws -> FloatTensor {
        try validate(input, gamma: gamma, epsilon: epsilon)
        let rows = input.shape[0]
        let width = input.shape[1]
        var output = Array(repeating: Float.zero, count: input.elementCount)
        for row in 0..<rows {
            var sumSquares: Float = 0
            for column in 0..<width {
                let value = input.storage[row * width + column]
                sumSquares += value * value
            }
            let inverseRMS = 1 / sqrt(sumSquares / Float(width) + epsilon)
            for column in 0..<width {
                output[row * width + column] = input.storage[row * width + column] * inverseRMS * gamma.storage[column]
            }
        }
        return try FloatTensor(output, shape: input.shape)
    }

    private static func validate(_ input: FloatTensor, gamma: FloatTensor, epsilon: Float) throws {
        guard input.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: input.rank) }
        guard gamma.rank == 1 else { throw TensorError.rankMismatch(expected: 1, actual: gamma.rank) }
        guard input.shape[1] > 0 else { throw RMSNormError.emptyFeatureWidth }
        guard gamma.shape[0] == input.shape[1] else { throw RMSNormError.gammaWidthMismatch(expected: input.shape[1], actual: gamma.shape[0]) }
        guard epsilon.isFinite, epsilon > 0 else { throw RMSNormError.invalidEpsilon(epsilon) }
        for row in 0..<input.shape[0] {
            for column in 0..<input.shape[1] where !input.storage[row * input.shape[1] + column].isFinite {
                throw RMSNormError.nonFiniteInput(row: row, column: column)
            }
        }
    }
}