import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P003TransposeMetalTests: XCTestCase {
    func testCanonicalMetalSolutionPassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is unavailable on this test host.")
        }
        let pipeline = try P003TransposeSolution.makeMetalPipeline()
        let report = P003TransposeJudge.evaluate(pipeline.transpose)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}