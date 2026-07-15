import InferenceSchoolCore

public enum P005GEMMSolution {
    public static func multiply(_ lhs: FloatTensor, _ rhs: FloatTensor) throws -> FloatTensor {
        guard lhs.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: lhs.rank)
        }
        guard rhs.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: rhs.rank)
        }
        let m = lhs.shape[0]
        let k = lhs.shape[1]
        let n = rhs.shape[1]
        guard k == rhs.shape[0] else {
            throw DenseLinearAlgebraError.innerDimensionMismatch(
                operation: "GEMM",
                lhs: k,
                rhs: rhs.shape[0]
            )
        }

        var output = Array(repeating: Float.zero, count: m * n)
        for row in 0..<m {
            for column in 0..<n {
                var sum: Float = 0
                for inner in 0..<k {
                    sum += lhs.storage[row * k + inner] * rhs.storage[inner * n + column]
                }
                output[row * n + column] = sum
            }
        }
        return try FloatTensor(output, shape: [m, n])
    }
}