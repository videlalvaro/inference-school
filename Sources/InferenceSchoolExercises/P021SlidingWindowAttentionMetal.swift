import Foundation
import InferenceSchoolCore

extension P021SlidingWindowAttentionExercise {
  public static func makeMetalPipeline() throws -> MetalStreamingAttentionPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P021SlidingWindowAttention", withExtension: "metal", subdirectory: "Metal")
    else {
      throw MetalNeuralOperatorError.kernelResourceMissing("P021SlidingWindowAttention.metal")
    }
    return try MetalStreamingAttentionPipeline(
      source: String(contentsOf: url, encoding: .utf8), functionName: "sliding_window_attention",
      operation: "sliding-window attention")
  }
}
