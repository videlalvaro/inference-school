import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P012FusedRMSNormGEMVMetalTests: XCTestCase {
    func testCanonicalFusedMetalSolutionPassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable on this test host.") }
        let pipeline = try P012FusedRMSNormGEMVSolution.makeFusedMetalPipeline()
        let report = P012FusedRMSNormGEMVJudge.evaluate(pipeline.project)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}