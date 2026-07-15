import Foundation
import InferenceSchoolCore

extension P033FusedQ4GEMVSolution {
  public static func makeMetalPipeline() throws -> MetalFusedQ4GEMVPipeline {
    guard let url = Bundle.module.url(
      forResource: "P033FusedQ4GEMV", withExtension: "metal", subdirectory: "Metal")
    else { throw MetalNeuralOperatorError.kernelResourceMissing("P033FusedQ4GEMV.metal") }
    return try MetalFusedQ4GEMVPipeline(source: String(contentsOf: url, encoding: .utf8))
  }
}