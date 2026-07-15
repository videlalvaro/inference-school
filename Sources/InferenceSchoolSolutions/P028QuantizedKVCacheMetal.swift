import Foundation
import InferenceSchoolCore

extension P028QuantizedKVCacheSolution {
  public static func makeMetalPipeline() throws -> MetalQuantizedCachedAttentionPipeline {
    guard let url = Bundle.module.url(
      forResource: "P028QuantizedKVCache", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P028QuantizedKVCache.metal") }
    return try MetalQuantizedCachedAttentionPipeline(
      source: String(contentsOf: url, encoding: .utf8))
  }
}
