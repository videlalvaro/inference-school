import Foundation
import InferenceSchoolCore

extension P023CachedAttentionExercise {
  public static func makeMetalPipeline() throws -> MetalCachedAttentionPipeline {
    guard let url = Bundle.module.url(
      forResource: "P023CachedAttention", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P023CachedAttention.metal") }
    return try MetalCachedAttentionPipeline(source: String(contentsOf: url, encoding: .utf8))
  }

  public static func runMetal(
    _ request: CachedAttentionRequest,
    pipeline: MetalCachedAttentionPipeline
  ) throws -> CachedAttentionResult {
    let zeroStorage = Array(repeating: Float.zero, count: request.cacheConfiguration.elementsPerTensor)
    return CachedAttentionResult(
      output: try pipeline.apply(
        query: request.query,
        keyStorage: zeroStorage,
        valueStorage: zeroStorage,
        configuration: request.cacheConfiguration,
        queryHeadCount: request.attentionConfiguration.queryHeadCount,
        layer: request.layer,
        tokenCount: request.tokenCount),
      cachedLogicalPositions: Array(request.firstLogicalPosition...request.queryLogicalPosition),
      allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes)
  }
}
