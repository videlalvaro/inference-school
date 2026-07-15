import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P010RMSNormMetalTests: XCTestCase {
    func testCanonicalMetalSolutionPassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable on this test host.") }
        let pipeline = try P010RMSNormSolution.makeMetalPipeline()
        let report = P010RMSNormJudge.evaluate(pipeline.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}