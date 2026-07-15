import InferenceSchoolCore

public enum P016CausalAttentionExercise {
  public static func apply(
    _ queries: FloatTensor, _ keys: FloatTensor, _ values: FloatTensor,
    _ configuration: AttentionConfiguration
  ) throws -> FloatTensor {
    _ = try P016CausalAttentionContract.validate(
      queries: queries, keys: keys, values: values, configuration: configuration)
    return try FloatTensor(Array(repeating: 0, count: queries.elementCount), shape: queries.shape)
  }
}
