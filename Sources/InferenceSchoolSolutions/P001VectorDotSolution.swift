import InferenceSchoolCore

public enum P001VectorDotSolution {
    public static func dot(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
        guard lhs.count == rhs.count else {
            throw VectorDotError.lengthMismatch(lhs: lhs.count, rhs: rhs.count)
        }

        var result: Float = 0
        for index in lhs.indices {
            result += lhs[index] * rhs[index]
        }
        return result
    }
}