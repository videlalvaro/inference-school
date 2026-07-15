import InferenceSchoolCore

public enum P001VectorDotExercise {
    public static func dot(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
        guard lhs.count == rhs.count else {
            throw VectorDotError.lengthMismatch(lhs: lhs.count, rhs: rhs.count)
        }

        // TODO: Replace this placeholder with the CPU reference implementation.
        return 0
    }
}