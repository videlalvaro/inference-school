import Foundation
import InferenceSchoolCore

extension P015RoPEExercise {
  public static func makeMetalPipeline() throws -> MetalRoPEPipeline {
    guard
      let url = Bundle.module.url(
        forResource: "P015RoPE", withExtension: "metal", subdirectory: "Metal")
    else {
      throw MetalNeuralOperatorError.kernelResourceMissing("P015RoPE.metal")
    }
    return try MetalRoPEPipeline(source: String(contentsOf: url, encoding: .utf8))
  }
}
