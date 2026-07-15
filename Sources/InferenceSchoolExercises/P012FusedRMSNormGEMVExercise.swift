import InferenceSchoolCore

public enum P012FusedRMSNormGEMVExercise {
    public static func project(_ input: FloatTensor, gamma: FloatTensor, weights: FloatTensor, epsilon: Float) throws -> FloatTensor {
        try validate(input, gamma: gamma, weights: weights, epsilon: epsilon)
        // TODO: Compute the baseline RMSNorm and projection without changing the public contract.
        return try FloatTensor(Array(repeating: 0, count: weights.shape[0]), shape: [weights.shape[0]])
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