import Foundation
import InferenceSchoolCore

extension P003TransposeExercise {
    public static func makeMetalPipeline() throws -> MetalTransposePipeline {
        guard let url = Bundle.module.url(
            forResource: "P003Transpose",
            withExtension: "metal",
            subdirectory: "Metal"
        ) else {
            throw MetalTransposeError.kernelResourceMissing("P003Transpose.metal")
        }
        return try MetalTransposePipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}