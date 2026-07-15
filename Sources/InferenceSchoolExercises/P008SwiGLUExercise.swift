import InferenceSchoolCore

public enum P008SwiGLUExercise {
    public static func apply(_ input: FloatTensor, gateWeights: FloatTensor, upWeights: FloatTensor, downWeights: FloatTensor) throws -> FloatTensor {
        try validate(input, gateWeights: gateWeights, upWeights: upWeights, downWeights: downWeights)
        // TODO: Compute gate/up projections, SiLU(gate) * up, then the down projection.
        return try FloatTensor(Array(repeating: 0, count: downWeights.shape[0]), shape: [downWeights.shape[0]])
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