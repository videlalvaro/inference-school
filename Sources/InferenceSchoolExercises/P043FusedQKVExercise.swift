import InferenceSchoolCore

public enum P043FusedQKVExercise {
  public static func fused(_ request: FusedQKVRequest) throws -> FusedQKVResult {
    try P043FusedQKVContract.validate(request)
    let sequence = request.input.shape[0]
    let headDimension = request.configuration.headDimension
    // TODO: compute RMSNorm and all three projections without allocating a normalized tensor.
    return FusedQKVResult(
      queries: try FloatTensor(
        Array(repeating: 0, count: sequence * request.configuration.queryProjectionDimension),
        shape: [sequence, request.configuration.queryHeadCount, headDimension]),
      keys: try FloatTensor(
        Array(repeating: 0, count: sequence * request.configuration.keyValueProjectionDimension),
        shape: [sequence, request.configuration.keyValueHeadCount, headDimension]),
      values: try FloatTensor(
        Array(repeating: 0, count: sequence * request.configuration.keyValueProjectionDimension),
        shape: [sequence, request.configuration.keyValueHeadCount, headDimension]))
  }
}