import Foundation
import InferenceSchoolCore

public enum P020TiledAttentionSolution {
  public static func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration
  ) throws -> FloatTensor {
    let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
    try validateVisibleKeys(input)
    var output = Array(repeating: Float.zero, count: q.elementCount)
    let scale = 1 / sqrt(Float(c.headDimension))
    for query in 0..<input.queryLength {
      let qPosition = c.queryPositionOffset + query
      for qHead in 0..<c.queryHeadCount {
        let kvHead = c.keyValueHead(forQueryHead: qHead)
        var m = -Float.infinity
        var l: Float = 0
        var accumulator = Array(repeating: Float.zero, count: c.headDimension)
        for tileStart in stride(
          from: 0, to: input.keyValueLength, by: MetalTiledAttentionPipeline.keyTileSize)
        {
          let tileEnd = min(
            tileStart + MetalTiledAttentionPipeline.keyTileSize, input.keyValueLength)
          var scores: [(Int, Float)] = []
          for key in tileStart..<tileEnd where c.keyPositionOffset + key <= qPosition {
            var score: Float = 0
            for d in 0..<c.headDimension {
              score +=
                q.storage[input.queryOffset(sequence: query, head: qHead, feature: d)]
                * k.storage[input.keyValueOffset(sequence: key, head: kvHead, feature: d)]
            }
            scores.append((key, score * scale))
          }
          guard let tileMaximum = scores.map(\.1).max() else { continue }
          let newM = max(m, tileMaximum)
          let alpha = m.isFinite ? exp(m - newM) : 0
          for d in 0..<c.headDimension { accumulator[d] *= alpha }
          var tileSum: Float = 0
          for (key, score) in scores {
            let weight = exp(score - newM)
            tileSum += weight
            for d in 0..<c.headDimension {
              accumulator[d] +=
                weight * v.storage[input.keyValueOffset(sequence: key, head: kvHead, feature: d)]
            }
          }
          l = l * alpha + tileSum
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
