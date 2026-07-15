import Foundation
import InferenceSchoolCore

extension P017MultiHeadAttentionExercise {
  public static func makeMetalPipeline() throws -> MetalParallelAttentionPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P017MultiHeadAttention", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P017MultiHeadAttention.metal") }
    return try MetalParallelAttentionPipeline(
      source: String(contentsOf: url, encoding: .utf8), functionName: "multi_head_attention",
      operation: "multi-head attention")
  }
}
