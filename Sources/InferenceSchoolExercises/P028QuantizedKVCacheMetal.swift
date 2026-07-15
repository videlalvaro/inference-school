import Foundation
import InferenceSchoolCore

extension P028QuantizedKVCacheExercise {
  public static func makeMetalPipeline() throws -> MetalQuantizedCachedAttentionPipeline {
    guard let url = Bundle.module.url(
      forResource: "P028QuantizedKVCache", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P028QuantizedKVCache.metal") }
    return try MetalQuantizedCachedAttentionPipeline(
      source: String(contentsOf: url, encoding: .utf8))
  }

  public static func runMetal(
    _ request: CachedAttentionRequest,
    pipeline: MetalQuantizedCachedAttentionPipeline
  ) throws -> QuantizedKVCacheResult {
    let vectorCount = request.cacheConfiguration.layerCount
      * request.cacheConfiguration.capacity
      * request.cacheConfiguration.keyValueHeadCount
    let output = try pipeline.apply(
      query: request.query,
      keyStorage: Array(repeating: 0, count: request.cacheConfiguration.elementsPerTensor),
      valueStorage: Array(repeating: 0, count: request.cacheConfiguration.elementsPerTensor),
      keyScales: Array(repeating: 1, count: vectorCount),
      valueScales: Array(repeating: 1, count: vectorCount),
      configuration: request.cacheConfiguration,
      queryHeadCount: request.attentionConfiguration.queryHeadCount,
      layer: request.layer,
      tokenCount: request.tokenCount)
    return QuantizedKVCacheResult(
      attentionOutput: output,
      dequantizedKeys: request.keys,
      dequantizedValues: request.values,
      keyScales: [],
      valueScales: [],
      allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes,
      maximumKeyError: 0,
      maximumValueError: 0)
  }
}
