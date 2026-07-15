import Foundation
import InferenceSchoolCore

public enum P009SoftmaxSolution {
    public static func apply(_ logits: FloatTensor) throws -> FloatTensor {
        try validate(logits)
        let rows = logits.shape[0]
        let columns = logits.shape[1]
        var output = Array(repeating: Float.zero, count: logits.elementCount)
        for row in 0..<rows {
            let start = row * columns
            var maximum = -Float.infinity
            for column in 0..<columns { maximum = max(maximum, logits.storage[start + column]) }
            var sum: Float = 0
            for column in 0..<columns {
                output[start + column] = exp(logits.storage[start + column] - maximum)
                sum += output[start + column]
            }
            for column in 0..<columns { output[start + column] /= sum }
        }
        return try FloatTensor(output, shape: logits.shape)
    }

    private static func validate(_ logits: FloatTensor) throws {
        guard logits.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: logits.rank) }
        guard logits.shape[1] > 0 else { throw SoftmaxError.emptyRow }
        for row in 0..<logits.shape[0] {
            for column in 0..<logits.shape[1] where !logits.storage[row * logits.shape[1] + column].isFinite {
                throw SoftmaxError.nonFiniteInput(row: row, column: column)
            }
        }
    }
}