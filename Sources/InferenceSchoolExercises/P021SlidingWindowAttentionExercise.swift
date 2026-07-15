import InferenceSchoolCore

public enum P021SlidingWindowAttentionExercise {
  public static func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration, _ window: Int
  ) throws -> FloatTensor {
    let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
    try validateVisibleKeys(input, window: window)
    return try FloatTensor(Array(repeating: 0, count: q.elementCount), shape: q.shape)
  }
}
