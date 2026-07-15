import Foundation
import InferenceSchoolCore

extension P009SoftmaxSolution {
    public static func makeMetalPipeline() throws -> MetalSoftmaxPipeline {
        guard let url = Bundle.module.url(forResource: "P009Softmax", withExtension: "metal", subdirectory: "Metal") else {
            throw MetalNeuralOperatorError.kernelResourceMissing("P009Softmax.metal")
        }
        return try MetalSoftmaxPipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}