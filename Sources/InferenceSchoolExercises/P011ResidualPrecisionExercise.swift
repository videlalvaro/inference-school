import InferenceSchoolCore

public enum P011ResidualPrecisionExercise {
    public static func accumulate(_ initial: FloatTensor, updates: [FloatTensor], policy: ResidualPrecisionPolicy) throws -> FloatTensor {
        for (index, update) in updates.enumerated() where update.shape != initial.shape {
            throw ResidualStreamError.updateShapeMismatch(index: index, expected: initial.shape, actual: update.shape)
        }
        // TODO: Add every update and apply the selected storage policy at the stated boundary.
        return initial
    }
}