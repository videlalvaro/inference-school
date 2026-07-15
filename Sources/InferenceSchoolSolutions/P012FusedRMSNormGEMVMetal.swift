import Foundation
import InferenceSchoolCore

extension P012FusedRMSNormGEMVSolution {
    public static func makeFusedMetalPipeline() throws -> MetalFusedRMSNormGEMVPipeline {
        guard let url = Bundle.module.url(forResource: "P012FusedRMSNormGEMV", withExtension: "metal", subdirectory: "Metal") else {
            throw MetalNeuralOperatorError.kernelResourceMissing("P012FusedRMSNormGEMV.metal")
        }
        return try MetalFusedRMSNormGEMVPipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}