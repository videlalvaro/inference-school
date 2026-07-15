import Foundation
import InferenceSchoolCore

extension P023CachedAttentionSolution {
  public static func makeMetalPipeline() throws -> MetalCachedAttentionPipeline {
    guard let url = Bundle.module.url(
      forResource: "P023CachedAttention", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P023CachedAttention.metal") }
    return try MetalCachedAttentionPipeline(source: String(contentsOf: url, encoding: .utf8))
  }
}
