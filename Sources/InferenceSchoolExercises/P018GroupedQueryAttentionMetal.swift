import Foundation
import InferenceSchoolCore

extension P018GroupedQueryAttentionExercise {
  public static func makeMetalPipeline() throws -> MetalParallelAttentionPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P018GroupedQueryAttention", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P018GroupedQueryAttention.metal") }
    return try MetalParallelAttentionPipeline(
      source: String(contentsOf: url, encoding: .utf8), functionName: "grouped_query_attention",
      operation: "grouped-query attention")
  }
}
