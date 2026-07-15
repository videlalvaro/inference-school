import Foundation
import InferenceSchoolCore

extension P010RMSNormExercise {
    public static func makeMetalPipeline() throws -> MetalRMSNormPipeline {
        guard let url = Bundle.module.url(forResource: "P010RMSNorm", withExtension: "metal", subdirectory: "Metal") else {
            throw MetalNeuralOperatorError.kernelResourceMissing("P010RMSNorm.metal")
        }
        return try MetalRMSNormPipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}