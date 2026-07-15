import Foundation
import InferenceSchoolCore

public enum P016CausalAttentionSolution {
  public static func apply(
    _ queries: FloatTensor, _ keys: FloatTensor, _ values: FloatTensor,
    _ configuration: AttentionConfiguration
  ) throws -> FloatTensor {
    let input = try P016CausalAttentionContract.validate(
      queries: queries, keys: keys, values: values, configuration: configuration)
    var scores = Array(repeating: -Float.infinity, count: input.queryLength * input.keyValueLength)
    let scale = 1 / sqrt(Float(configuration.headDimension))
    for query in 0..<input.queryLength {
      let queryPosition = configuration.queryPositionOffset + query
      for key in 0..<input.keyValueLength
      where configuration.keyPositionOffset + key <= queryPosition {
        var dot: Float = 0
        for feature in 0..<configuration.headDimension {
          dot +=
            queries.storage[input.queryOffset(sequence: query, head: 0, feature: feature)]
            * keys.storage[input.keyValueOffset(sequence: key, head: 0, feature: feature)]
        }
        scores[query * input.keyValueLength + key] = dot * scale
      }
    }
    var output = Array(repeating: Float.zero, count: queries.elementCount)
    for query in 0..<input.queryLength {
      let rowStart = query * input.keyValueLength
      let row = scores[rowStart..<(rowStart + input.keyValueLength)]
      let maximum = row.max()!
      let exponentials = row.map { exp($0 - maximum) }
      let denominator = exponentials.reduce(0, +)
      for feature in 0..<configuration.headDimension {
        var sum: Float = 0
        for key in 0..<input.keyValueLength where scores[rowStart + key].isFinite {
          sum +=
            exponentials[key] / denominator
            * values.storage[input.keyValueOffset(sequence: key, head: 0, feature: feature)]
        }
        output[input.queryOffset(sequence: query, head: 0, feature: feature)] = sum
      }
    }
    return try FloatTensor(output, shape: queries.shape)
  }
}
