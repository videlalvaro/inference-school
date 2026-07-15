import Foundation
import InferenceSchoolCore

extension P013EmbeddingExercise {
  public static func makeMetalPipeline() throws -> MetalEmbeddingPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P013Embedding",
        withExtension: "metal",
        subdirectory: "Metal"
      )
    else {
      throw MetalNeuralOperatorError.kernelResourceMissing("P013Embedding.metal")
    }
    return try MetalEmbeddingPipeline(source: String(contentsOf: url, encoding: .utf8))
  }
}
