import InferenceSchoolCore

public enum P014QKVProjectionSolution {
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
    func projectOne(_ weights: FloatTensor, heads: Int) throws -> FloatTensor {
      let sequenceLength = hidden.shape[0]
      let modelDimension = hidden.shape[1]
      let columns = heads * configuration.headDimension
      var output = Array(repeating: Float.zero, count: sequenceLength * columns)
      for sequence in 0..<sequenceLength {
        for head in 0..<heads {
          for feature in 0..<configuration.headDimension {
            let column = head * configuration.headDimension + feature
            var sum: Float = 0
            for model in 0..<modelDimension {
              sum +=
                hidden.storage[sequence * modelDimension + model]
                * weights.storage[model * columns + column]
            }
            output[sequence * columns + column] = sum
          }
        }
      }
      return try FloatTensor(
        output,
        shape: [sequenceLength, heads, configuration.headDimension]
      )
    }
    return QKVProjectionResult(
      queries: try projectOne(queryWeights, heads: configuration.queryHeadCount),
      keys: try projectOne(keyWeights, heads: configuration.keyValueHeadCount),
      values: try projectOne(valueWeights, heads: configuration.keyValueHeadCount)
    )
  }
}
