import Foundation
import InferenceSchoolCore

extension P016CausalAttentionSolution {
  public static func makeMetalPipeline() throws -> MetalMaterializedAttentionPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P016CausalAttention", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P016CausalAttention.metal") }
    return try MetalMaterializedAttentionPipeline(source: String(contentsOf: url, encoding: .utf8))
  }
}
