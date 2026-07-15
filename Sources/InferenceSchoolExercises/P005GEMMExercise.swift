import InferenceSchoolCore

public enum P005GEMMExercise {
    public static func multiply(_ lhs: FloatTensor, _ rhs: FloatTensor) throws -> FloatTensor {
        guard lhs.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: lhs.rank)
        }
        guard rhs.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: rhs.rank)
        }
        guard lhs.shape[1] == rhs.shape[0] else {
            throw DenseLinearAlgebraError.innerDimensionMismatch(
                operation: "GEMM",
                lhs: lhs.shape[1],
                rhs: rhs.shape[0]
            )
        }

        // TODO: Accumulate each output cell over the shared inner dimension.
        let shape = [lhs.shape[0], rhs.shape[1]]
        return try FloatTensor(Array(repeating: 0, count: shape[0] * shape[1]), shape: shape)
    }
}