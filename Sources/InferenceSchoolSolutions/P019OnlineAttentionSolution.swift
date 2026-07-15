import Foundation
import InferenceSchoolCore

public enum P019OnlineAttentionSolution {
  public static func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration
  ) throws -> FloatTensor {
    let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
    try validateVisibleKeys(input)
    var output = Array(repeating: Float.zero, count: q.elementCount)
    let scale = 1 / sqrt(Float(c.headDimension))
    for query in 0..<input.queryLength {
      let queryPosition = c.queryPositionOffset + query
      for queryHead in 0..<c.queryHeadCount {
        let kvHead = c.keyValueHead(forQueryHead: queryHead)
        var maximum = -Float.infinity
        var denominator: Float = 0
        var accumulator = Array(repeating: Float.zero, count: c.headDimension)
        for key in 0..<input.keyValueLength where c.keyPositionOffset + key <= queryPosition {
          var score: Float = 0
          for feature in 0..<c.headDimension {
            score +=
              q.storage[input.queryOffset(sequence: query, head: queryHead, feature: feature)]
              * k.storage[input.keyValueOffset(sequence: key, head: kvHead, feature: feature)]
          }
          score *= scale
          let newMaximum = max(maximum, score)
          let previousScale = maximum.isFinite ? exp(maximum - newMaximum) : 0
          let weight = exp(score - newMaximum)
          denominator = denominator * previousScale + weight
          for feature in 0..<c.headDimension {
            accumulator[feature] =
              accumulator[feature] * previousScale + weight
              * v.storage[input.keyValueOffset(sequence: key, head: kvHead, feature: feature)]
          }
          maximum = newMaximum
        }
        for feature in 0..<c.headDimension {
          output[input.queryOffset(sequence: query, head: queryHead, feature: feature)] =
            accumulator[feature] / denominator
        }
      }
    }
    return try FloatTensor(output, shape: q.shape)
  }
}
