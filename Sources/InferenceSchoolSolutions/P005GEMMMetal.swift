import Foundation
import InferenceSchoolCore

extension P005GEMMSolution {
    public static func makeMetalPipeline() throws -> MetalGEMMPipeline {
        guard let url = Bundle.module.url(
            forResource: "P005GEMM",
            withExtension: "metal",
            subdirectory: "Metal"
        ) else {
            throw MetalGEMMError.kernelResourceMissing("P005GEMM.metal")
        }
        return try MetalGEMMPipeline(source: String(contentsOf: url, encoding: .utf8))
    }
}