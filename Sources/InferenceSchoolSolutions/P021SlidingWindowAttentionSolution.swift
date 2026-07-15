import Foundation
import InferenceSchoolCore

public enum P021SlidingWindowAttentionSolution {
  public static func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration, _ window: Int
  ) throws -> FloatTensor {
    let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
    try validateVisibleKeys(input, window: window)
    var output = Array(repeating: Float.zero, count: q.elementCount)
    let scale = 1 / sqrt(Float(c.headDimension))
    for query in 0..<input.queryLength {
      let qPosition = c.queryPositionOffset + query
      let lower = qPosition - window + 1
      for qHead in 0..<c.queryHeadCount {
        let kvHead = c.keyValueHead(forQueryHead: qHead)
        var m = -Float.infinity
        var l: Float = 0
        var accumulator = Array(repeating: Float.zero, count: c.headDimension)
        for key in 0..<input.keyValueLength {
          let keyPosition = c.keyPositionOffset + key
          guard keyPosition >= lower, keyPosition <= qPosition else { continue }
          var score: Float = 0
          for d in 0..<c.headDimension {
            score +=
              q.storage[input.queryOffset(sequence: query, head: qHead, feature: d)]
              * k.storage[input.keyValueOffset(sequence: key, head: kvHead, feature: d)]
          }
          score *= scale
          let newM = max(m, score)
          let alpha = m.isFinite ? exp(m - newM) : 0
          let beta = exp(score - newM)
          l = l * alpha + beta
          for d in 0..<c.headDimension {
            accumulator[d] =
              accumulator[d] * alpha + beta
              * v.storage[input.keyValueOffset(sequence: key, head: kvHead, feature: d)]
          }
          m = newM
        }
        for d in 0..<c.headDimension {
          output[input.queryOffset(sequence: query, head: qHead, feature: d)] = accumulator[d] / l
        }
      }
    }
    return try FloatTensor(output, shape: q.shape)
  }
}
