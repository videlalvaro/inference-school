import Foundation
import InferenceSchoolCore

extension P004GEMVSolution {
    public static func makeMetalPipeline() throws -> MetalGEMVPipeline {
        guard let url = Bundle.module.url(
            forResource: "P004GEMV",
            withExtension: "metal",
            subdirectory: "Metal"
        ) else {
            throw MetalGEMVError.kernelResourceMissing("P004GEMV.metal")
        }
        return try MetalGEMVPipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}