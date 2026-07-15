import InferenceSchoolCore

public enum P009SoftmaxExercise {
    public static func apply(_ logits: FloatTensor) throws -> FloatTensor {
        guard logits.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: logits.rank) }
        guard logits.shape[1] > 0 else { throw SoftmaxError.emptyRow }
        for row in 0..<logits.shape[0] {
            for column in 0..<logits.shape[1] where !logits.storage[row * logits.shape[1] + column].isFinite {
                throw SoftmaxError.nonFiniteInput(row: row, column: column)
            }
        }
        // TODO: Subtract each row maximum before exponentiation, then normalize.
        return try FloatTensor(Array(repeating: 0, count: logits.elementCount), shape: logits.shape)
    }
}