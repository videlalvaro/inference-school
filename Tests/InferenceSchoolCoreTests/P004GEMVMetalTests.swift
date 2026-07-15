import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P004GEMVMetalTests: XCTestCase {
    func testCanonicalMetalSolutionPassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is unavailable on this test host.")
        }
        let pipeline = try P004GEMVSolution.makeMetalPipeline()
        let report = P004GEMVJudge.evaluate(pipeline.multiply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}