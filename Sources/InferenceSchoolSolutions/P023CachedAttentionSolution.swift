import Foundation
import InferenceSchoolCore

public enum P023CachedAttentionSolution {
  public static func run(_ request: CachedAttentionRequest) throws -> CachedAttentionResult {
    let cache = try populatedCache(request)
    return CachedAttentionResult(
      output: try attend(
        query: request.query,
        cache: cache,
        layer: request.layer,
        queryLogicalPosition: request.queryLogicalPosition,
        queryHeadCount: request.attentionConfiguration.queryHeadCount),
      cachedLogicalPositions: try cache.logicalPositions(layer: request.layer),
      allocatedBytes: cache.allocatedBytes)
  }

  public static func runMetal(
    _ request: CachedAttentionRequest,
    pipeline: MetalCachedAttentionPipeline
  ) throws -> CachedAttentionResult {
    let cache = try populatedCache(request)
    return CachedAttentionResult(
      output: try pipeline.apply(
        query: request.query,
        keyStorage: cache.rawKeyStorage(),
        valueStorage: cache.rawValueStorage(),
        configuration: cache.configuration,
        queryHeadCount: request.attentionConfiguration.queryHeadCount,
        layer: request.layer,
        tokenCount: try cache.count(layer: request.layer)),
      cachedLogicalPositions: try cache.logicalPositions(layer: request.layer),
      allocatedBytes: cache.allocatedBytes)
  }

  public static func attend(
    query: FloatTensor,
    cache: any KVCacheReadable,
    layer: Int,
    queryLogicalPosition: Int,
    queryHeadCount: Int,
    window: Int? = nil
  ) throws -> FloatTensor {
    try cache.configuration.validate(layer: layer)
    let configuration = try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: cache.configuration.keyValueHeadCount,
      headDimension: cache.configuration.headDimension)
    let expected = [queryHeadCount, cache.configuration.headDimension]
    guard query.shape == expected else {
      throw KVCacheError.vectorShapeMismatch(name: "Query", expected: expected, actual: query.shape)
    }
    if let window, window <= 0 { throw AttentionError.invalidWindow(window) }
    let positions = try cache.logicalPositions(layer: layer).filter { position in
      position <= queryLogicalPosition
        && (window == nil || position >= queryLogicalPosition - window! + 1)
    }
    guard !positions.isEmpty else {
      throw AttentionError.noVisibleKeys(queryPosition: queryLogicalPosition)
    }
    var output = Array(repeating: Float.zero, count: query.elementCount)
    let scale = 1 / sqrt(Float(configuration.headDimension))
    for queryHead in 0..<queryHeadCount {
      let keyValueHead = configuration.keyValueHead(forQueryHead: queryHead)
      var maximum = -Float.infinity
      var denominator: Float = 0
      var accumulator = Array(repeating: Float.zero, count: configuration.headDimension)
      for position in positions {
        let key = try cache.keyVector(
          layer: layer, logicalPosition: position, head: keyValueHead)
        let value = try cache.valueVector(
          layer: layer, logicalPosition: position, head: keyValueHead)
        var score: Float = 0
        for feature in 0..<configuration.headDimension {
          score += query.storage[queryHead * configuration.headDimension + feature] * key[feature]
        }
        score *= scale
        let nextMaximum = max(maximum, score)
        let alpha = maximum.isFinite ? exp(maximum - nextMaximum) : 0
        let beta = exp(score - nextMaximum)
        denominator = denominator * alpha + beta
        for feature in 0..<configuration.headDimension {
          accumulator[feature] = accumulator[feature] * alpha + beta * value[feature]
        }
        maximum = nextMaximum
      }
      for feature in 0..<configuration.headDimension {
        output[queryHead * configuration.headDimension + feature]
          = accumulator[feature] / denominator
      }
    }
    return try FloatTensor(output, shape: query.shape)
  }

  private static func populatedCache(_ request: CachedAttentionRequest) throws -> ContiguousKVCache {
    let cache = ContiguousKVCache(configuration: request.cacheConfiguration)
    let elements = request.cacheConfiguration.elementsPerToken
    for token in 0..<request.tokenCount {
      let start = token * elements
      let end = start + elements
      try cache.append(
        layer: request.layer,
        logicalPosition: request.firstLogicalPosition + token,
        key: FloatTensor(
          Array(request.keys.storage[start..<end]),
          shape: [request.cacheConfiguration.keyValueHeadCount,
            request.cacheConfiguration.headDimension]),
        value: FloatTensor(
          Array(request.values.storage[start..<end]),
          shape: [request.cacheConfiguration.keyValueHeadCount,
            request.cacheConfiguration.headDimension]))
    }
    return cache
  }
}
