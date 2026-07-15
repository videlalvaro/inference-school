import Foundation
import InferenceSchoolCore

extension P019OnlineAttentionExercise {
  public static func makeMetalPipeline() throws -> MetalStreamingAttentionPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P019OnlineAttention", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P019OnlineAttention.metal") }
    return try MetalStreamingAttentionPipeline(
      source: String(contentsOf: url, encoding: .utf8), functionName: "online_attention",
      operation: "online attention")
  }
}
