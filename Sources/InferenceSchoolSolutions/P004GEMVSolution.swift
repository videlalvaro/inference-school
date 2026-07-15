import InferenceSchoolCore

public enum P004GEMVSolution {
    public static func multiply(matrix: FloatTensor, vector: FloatTensor) throws -> FloatTensor {
        guard matrix.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: matrix.rank)
        }
        guard vector.rank == 1 else {
            throw TensorError.rankMismatch(expected: 1, actual: vector.rank)
        }
        let rows = matrix.shape[0]
        let columns = matrix.shape[1]
        guard columns == vector.shape[0] else {
            throw DenseLinearAlgebraError.innerDimensionMismatch(
                operation: "GEMV",
                lhs: columns,
                rhs: vector.shape[0]
            )
        }

        var output = Array(repeating: Float.zero, count: rows)
        for row in 0..<rows {
            var sum: Float = 0
            for column in 0..<columns {
                sum += matrix.storage[row * columns + column] * vector.storage[column]
            }
            output[row] = sum
        }
        return try FloatTensor(output, shape: [rows])
    }
}