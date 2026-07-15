import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P001VectorDotTests: XCTestCase {
    func testCanonicalSolutionPassesTheJudge() {
        let report = P001VectorDotJudge.evaluate(P001VectorDotSolution.dot)

        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsAnIncorrectImplementation() {
        let report = P001VectorDotJudge.evaluate { _, _ in 0 }

        XCTAssertFalse(report.isPassing)
        XCTAssertFalse(report.failures.isEmpty)
    }

    func testCanonicalSolutionRejectsMismatchedLengths() {
        XCTAssertThrowsError(try P001VectorDotSolution.dot([1, 2], [1])) { error in
            XCTAssertEqual(
                error as? VectorDotError,
                .lengthMismatch(lhs: 2, rhs: 1)
            )
        }
    }
}