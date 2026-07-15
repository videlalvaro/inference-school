import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P009SoftmaxTests: XCTestCase {
    func testCanonicalCPUSolutionPassesJudge() {
        let report = P009SoftmaxJudge.evaluate(P009SoftmaxSolution.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsUniformProbabilities() {
        let report = P009SoftmaxJudge.evaluate { logits in
            guard logits.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: logits.rank) }
            guard logits.shape[1] > 0 else { throw SoftmaxError.emptyRow }
            let probability = 1 / Float(logits.shape[1])
            return try FloatTensor(Array(repeating: probability, count: logits.elementCount), shape: logits.shape)
        }
        XCTAssertFalse(report.isPassing)
    }
}