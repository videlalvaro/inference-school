import InferenceSchoolCore

public enum P010RMSNormExercise {
    public static func apply(_ input: FloatTensor, gamma: FloatTensor, epsilon: Float) throws -> FloatTensor {
        try validateRMSNorm(input, gamma: gamma, epsilon: epsilon)
        // TODO: Compute x / sqrt(mean(x^2) + epsilon) * gamma for every row.
        return try FloatTensor(Array(repeating: 0, count: input.elementCount), shape: input.shape)
    }

    private static func validateRMSNorm(_ input: FloatTensor, gamma: FloatTensor, epsilon: Float) throws {
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