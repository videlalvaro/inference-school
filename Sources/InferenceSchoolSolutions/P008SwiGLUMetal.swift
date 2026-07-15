import Foundation
import InferenceSchoolCore

extension P008SwiGLUSolution {
    public static func makeMetalGatePipeline() throws -> MetalSwiGLUGatePipeline {
        guard let url = Bundle.module.url(forResource: "P008SwiGLU", withExtension: "metal", subdirectory: "Metal") else {
            throw MetalNeuralOperatorError.kernelResourceMissing("P008SwiGLU.metal")
        }
        return try MetalSwiGLUGatePipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}