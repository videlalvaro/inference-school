import Foundation
import InferenceSchoolCore

extension P007ActivationExercise {
    public static func makeMetalPipeline() throws -> MetalActivationPipeline {
        guard let url = Bundle.module.url(forResource: "P007Activation", withExtension: "metal", subdirectory: "Metal") else {
            throw MetalNeuralOperatorError.kernelResourceMissing("P007Activation.metal")
        }
        return try MetalActivationPipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}