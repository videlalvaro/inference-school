import InferenceSchoolCore

public enum P003TransposeSolution {
    public static func transpose(_ input: FloatTensor) throws -> FloatTensor {
        guard input.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: input.rank)
        }
        let rows = input.shape[0]
        let columns = input.shape[1]
        var output = Array(repeating: Float.zero, count: input.elementCount)
        for row in 0..<rows {
            for column in 0..<columns {
                output[column * rows + row] = input.storage[row * columns + column]
            }
        }
        return try FloatTensor(output, shape: [columns, rows])
    }
}