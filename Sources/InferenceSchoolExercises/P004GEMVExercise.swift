import InferenceSchoolCore

public enum P004GEMVExercise {
    public static func multiply(matrix: FloatTensor, vector: FloatTensor) throws -> FloatTensor {
        guard matrix.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: matrix.rank)
        }
        guard vector.rank == 1 else {
            throw TensorError.rankMismatch(expected: 1, actual: vector.rank)
        }
        guard matrix.shape[1] == vector.shape[0] else {
            throw DenseLinearAlgebraError.innerDimensionMismatch(
                operation: "GEMV",
                lhs: matrix.shape[1],
                rhs: vector.shape[0]
            )
        }

        // TODO: Compute one matrix-row/vector dot product per output value.
        return try FloatTensor(Array(repeating: 0, count: matrix.shape[0]), shape: [matrix.shape[0]])
    }
}