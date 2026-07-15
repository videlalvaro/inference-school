import InferenceSchoolCore

public enum P036WeightContainerExercise {
  public static func parse(
    bytes: [UInt8],
    requiredTensorNames: [String]
  ) throws -> ParsedWeightContainer {
    _ = try P036WeightContainerContract.validatePreamble(bytes)
    _ = requiredTensorNames
    // TODO: decode and validate the JSON descriptors before reading any payload bytes.
    throw WeightContainerError.invalidJSONHeader("TODO: implement the InferenceWeight parser")
  }
}