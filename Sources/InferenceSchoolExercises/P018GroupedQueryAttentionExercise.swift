import InferenceSchoolCore

public enum P018GroupedQueryAttentionExercise {
  public static func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration
  ) throws -> FloatTensor {
    _ = try P018GroupedQueryAttentionContract.validate(q, k, v, c)
    return try FloatTensor(Array(repeating: 0, count: q.elementCount), shape: q.shape)
  }
}
