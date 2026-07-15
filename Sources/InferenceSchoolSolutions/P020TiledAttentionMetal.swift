import Foundation
import InferenceSchoolCore

extension P020TiledAttentionSolution {
  public static func makeMetalPipeline() throws -> MetalTiledAttentionPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P020TiledAttention", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P020TiledAttention.metal") }
    return try MetalTiledAttentionPipeline(source: String(contentsOf: url, encoding: .utf8))
  }
}
