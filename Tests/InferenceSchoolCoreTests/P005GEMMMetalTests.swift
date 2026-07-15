import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P005GEMMMetalTests: XCTestCase {
    func testCanonicalMetalSolutionPassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is unavailable on this test host.")
        }
        let pipeline = try P005GEMMSolution.makeMetalPipeline()
        let report = P005GEMMJudge.evaluate(pipeline.multiply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}