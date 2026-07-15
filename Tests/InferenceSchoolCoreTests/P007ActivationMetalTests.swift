import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P007ActivationMetalTests: XCTestCase {
    func testCanonicalMetalSolutionPassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable on this test host.") }
        let pipeline = try P007ActivationSolution.makeMetalPipeline()
        let report = P007ActivationJudge.evaluate(pipeline.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}