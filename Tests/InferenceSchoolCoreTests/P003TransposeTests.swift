import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P003TransposeTests: XCTestCase {
    func testCanonicalCPUSolutionPassesJudge() {
        let report = P003TransposeJudge.evaluate(P003TransposeSolution.transpose)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsShapeCorrectZeroOutput() {
        let report = P003TransposeJudge.evaluate { input in
            guard input.rank == 2 else {
                throw TensorError.rankMismatch(expected: 2, actual: input.rank)
            }
            return try FloatTensor(
                Array(repeating: 0, count: input.elementCount),
                shape: [input.shape[1], input.shape[0]]
            )
        }
        XCTAssertFalse(report.isPassing)
    }
}