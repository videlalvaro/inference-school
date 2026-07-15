import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P009SoftmaxMetalTests: XCTestCase {
    func testCanonicalMetalSolutionPassesJudge() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable on this test host.") }
        let pipeline = try P009SoftmaxSolution.makeMetalPipeline()
        let report = P009SoftmaxJudge.evaluate(pipeline.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testMetalRowWidthLimitIsExplicit() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable on this test host.") }
        let pipeline = try P009SoftmaxSolution.makeMetalPipeline()
        let logits = try FloatTensor(Array(repeating: 0, count: MetalSoftmaxPipeline.maximumRowWidth + 1), shape: [1, MetalSoftmaxPipeline.maximumRowWidth + 1])
        XCTAssertThrowsError(try pipeline.apply(logits)) { error in
            guard case MetalNeuralOperatorError.rowWidthExceedsMaximum(let maximum, let actual) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(maximum, 1024)
            XCTAssertEqual(actual, 1025)
        }
    }
}