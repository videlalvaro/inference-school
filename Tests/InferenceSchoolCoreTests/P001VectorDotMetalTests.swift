import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P001VectorDotMetalTests: XCTestCase {
    func testCanonicalMetalKernelPassesTheJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is unavailable on this test host.")
        }

        let pipeline = try P001VectorDotSolution.makeMetalPipeline()
        let report = P001VectorDotJudge.evaluate(pipeline.dot)

        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}