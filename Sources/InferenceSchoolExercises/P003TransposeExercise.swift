import InferenceSchoolCore

public enum P003TransposeExercise {
    public static func transpose(_ input: FloatTensor) throws -> FloatTensor {
        guard input.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: input.rank)
        }

        // TODO: Materialize the transpose in row-major output order.
        return try FloatTensor(
            Array(repeating: 0, count: input.elementCount),
            shape: [input.shape[1], input.shape[0]]
        )
    }
}