import Foundation
import InferenceSchoolCore

extension P001VectorDotSolution {
    public static func makeMetalPipeline() throws -> MetalVectorDotPipeline {
        guard let url = Bundle.module.url(
            forResource: "P001VectorDot",
            withExtension: "metal",
            subdirectory: "Metal"
        ) else {
            throw MetalVectorDotError.kernelResourceMissing("P001VectorDot.metal")
        }

        let source = try String(contentsOf: url, encoding: .utf8)
        return try MetalVectorDotPipeline(source: source)
    }
}