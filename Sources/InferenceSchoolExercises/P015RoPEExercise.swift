import InferenceSchoolCore

public enum P015RoPEExercise {
  public static func apply(
    queries: FloatTensor,
    keys: FloatTensor,
    rotaryDimension: Int,
    base: Float,
    queryPositionOffset: Int,
    keyPositionOffset: Int
  ) throws -> RoPEResult {
    try RoPEContract.validate(
      queries: queries, keys: keys, rotaryDimension: rotaryDimension, base: base,
      queryPositionOffset: queryPositionOffset, keyPositionOffset: keyPositionOffset)
    return RoPEResult(queries: queries, keys: keys)
  }
}
