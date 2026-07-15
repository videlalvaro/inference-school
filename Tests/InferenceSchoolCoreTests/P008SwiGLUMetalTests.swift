import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P008SwiGLUMetalTests: XCTestCase {
    func testCanonicalMetalGatePassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable on this test host.") }
        let pipeline = try P008SwiGLUSolution.makeMetalGatePipeline()
        let report = P008SwiGLUJudge.evaluateGate(pipeline.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }
}