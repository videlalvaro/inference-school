import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P004GEMVTests: XCTestCase {
    func testCanonicalCPUSolutionPassesJudge() {
        let report = P004GEMVJudge.evaluate(P004GEMVSolution.multiply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsValidatedZeroOutput() {
        let report = P004GEMVJudge.evaluate { matrix, vector in
            guard matrix.rank == 2 else {
                throw TensorError.rankMismatch(expected: 2, actual: matrix.rank)
            }
            guard vector.rank == 1 else {
                throw TensorError.rankMismatch(expected: 1, actual: vector.rank)
            }
            guard matrix.shape[1] == vector.shape[0] else {
                throw DenseLinearAlgebraError.innerDimensionMismatch(
                    operation: "GEMV",
                    lhs: matrix.shape[1],
                    rhs: vector.shape[0]
                )
            }
            return try FloatTensor(Array(repeating: 0, count: matrix.shape[0]), shape: [matrix.shape[0]])
        }
        XCTAssertFalse(report.isPassing)
    }
}