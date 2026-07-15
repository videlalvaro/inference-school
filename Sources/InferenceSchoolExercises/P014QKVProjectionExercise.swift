import InferenceSchoolCore

public enum P014QKVProjectionExercise {
  public static func project(
    hidden: FloatTensor,
    queryWeights: FloatTensor,
    keyWeights: FloatTensor,
    valueWeights: FloatTensor,
    configuration: AttentionConfiguration
  ) throws -> QKVProjectionResult {
    try QKVProjectionContract.validate(
      hidden: hidden,
      queryWeights: queryWeights,
      keyWeights: keyWeights,
      valueWeights: valueWeights,
      configuration: configuration
    )
    let sequenceLength = hidden.shape[0]
    return QKVProjectionResult(
      queries: try FloatTensor(
        Array(
          repeating: 0,
          count: sequenceLength * configuration.queryHeadCount * configuration.headDimension),
        shape: [sequenceLength, configuration.queryHeadCount, configuration.headDimension]
      ),
      keys: try FloatTensor(
        Array(
          repeating: 0,
          count: sequenceLength * configuration.keyValueHeadCount * configuration.headDimension),
        shape: [sequenceLength, configuration.keyValueHeadCount, configuration.headDimension]
      ),
      values: try FloatTensor(
        Array(
          repeating: 0,
          count: sequenceLength * configuration.keyValueHeadCount * configuration.headDimension),
        shape: [sequenceLength, configuration.keyValueHeadCount, configuration.headDimension]
      )
    )
  }
}
