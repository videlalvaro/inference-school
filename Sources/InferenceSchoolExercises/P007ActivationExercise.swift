import InferenceSchoolCore

public enum P007ActivationExercise {
    public static func apply(
        _ input: FloatTensor,
        activation: Activation
    ) throws -> FloatTensor {
        // TODO: Apply the selected activation independently to every value.
        try FloatTensor(input.storage, shape: input.shape)
    }
}