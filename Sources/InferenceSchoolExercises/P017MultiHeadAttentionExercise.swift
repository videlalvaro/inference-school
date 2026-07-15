import InferenceSchoolCore

public enum P017MultiHeadAttentionExercise {
  public static func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration
  ) throws -> FloatTensor {
    _ = try P017MultiHeadAttentionContract.validate(q, k, v, c)
    return try FloatTensor(Array(repeating: 0, count: q.elementCount), shape: q.shape)
  }
}
