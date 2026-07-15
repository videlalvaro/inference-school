import InferenceSchoolCore

public enum P023CachedAttentionExercise {
  public static func run(_ request: CachedAttentionRequest) throws -> CachedAttentionResult {
    CachedAttentionResult(
      output: try FloatTensor(
        Array(repeating: 0, count: request.query.elementCount), shape: request.query.shape),
      cachedLogicalPositions: Array(
        request.firstLogicalPosition...request.queryLogicalPosition),
      allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes)
  }
}
