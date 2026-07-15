import Foundation
import InferenceSchoolCore

extension P043FusedQKVExercise {
  public static func makeMetalPipeline() throws -> MetalFusedQKVPipeline {
    guard let url = Bundle.module.url(
      forResource: "P043FusedQKV", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P043FusedQKV.metal") }
    return try MetalFusedQKVPipeline(source: String(contentsOf: url, encoding: .utf8))
  }
}