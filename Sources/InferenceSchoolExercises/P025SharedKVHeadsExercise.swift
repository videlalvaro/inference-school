import Foundation
import InferenceSchoolCore

public enum P025SharedKVHeadsExercise {
  public static func run(_ request: CachedAttentionRequest) throws -> SharedKVHeadsResult {
    let c = request.attentionConfiguration
    var output = Array(repeating: Float.zero, count: request.query.elementCount)
    let scale = 1 / sqrt(Float(c.headDimension))
    for queryHead in 0..<c.queryHeadCount {
      let keyValueHead = queryHead % c.keyValueHeadCount
      var scores: [Float] = []
      for token in 0..<request.tokenCount {
        var score: Float = 0
        for feature in 0..<c.headDimension {
          score += request.query.storage[queryHead * c.headDimension + feature]
            * request.keys.storage[(token * c.keyValueHeadCount + keyValueHead)
              * c.headDimension + feature]
        }
        scores.append(score * scale)
      }
      let maximum = scores.max()!
      let weights = scores.map { exp($0 - maximum) }
      let denominator = weights.reduce(0, +)
      for feature in 0..<c.headDimension {
        for token in 0..<request.tokenCount {
          output[queryHead * c.headDimension + feature] += weights[token] / denominator
            * request.values.storage[(token * c.keyValueHeadCount + keyValueHead)
              * c.headDimension + feature]
        }
      }
    }
    return SharedKVHeadsResult(
      attention: CachedAttentionResult(
        output: try FloatTensor(output, shape: request.query.shape),
        cachedLogicalPositions: Array(
          request.firstLogicalPosition...request.queryLogicalPosition),
        allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes),
      bytes: try KVHeadMemoryModel.compare(
        layerCount: request.cacheConfiguration.layerCount,
        tokenCount: request.cacheConfiguration.capacity,
        queryHeadCount: c.queryHeadCount,
        gqaHeadCount: c.keyValueHeadCount,
        headDimension: c.headDimension))
  }
}
