import Foundation
import InferenceSchoolCore

public enum P017MultiHeadAttentionSolution {
  public static func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration
  ) throws -> FloatTensor {
    let input = try P017MultiHeadAttentionContract.validate(q, k, v, c)
    var output = Array(repeating: Float.zero, count: q.elementCount)
    let scale = 1 / sqrt(Float(c.headDimension))
    for query in 0..<input.queryLength {
      let queryPosition = c.queryPositionOffset + query
      for head in 0..<c.queryHeadCount {
        var scores: [Float] = []
        var visible: [Int] = []
        for key in 0..<input.keyValueLength where c.keyPositionOffset + key <= queryPosition {
          var dot: Float = 0
          for feature in 0..<c.headDimension {
            dot +=
              q.storage[input.queryOffset(sequence: query, head: head, feature: feature)]
              * k.storage[input.keyValueOffset(sequence: key, head: head, feature: feature)]
          }
          scores.append(dot * scale)
          visible.append(key)
        }
        let maximum = scores.max()!
        let weights = scores.map { exp($0 - maximum) }
        let denominator = weights.reduce(0, +)
        for feature in 0..<c.headDimension {
          var sum: Float = 0
          for index in visible.indices {
            sum +=
              weights[index] / denominator
              * v.storage[
                input.keyValueOffset(sequence: visible[index], head: head, feature: feature)]
          }
          output[input.queryOffset(sequence: query, head: head, feature: feature)] = sum
        }
      }
    }
    return try FloatTensor(output, shape: q.shape)
  }
}
