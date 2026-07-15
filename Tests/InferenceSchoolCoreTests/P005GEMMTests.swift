import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P005GEMMTests: XCTestCase {
    func testCanonicalCPUSolutionPassesJudge() {
        let report = P005GEMMJudge.evaluate(P005GEMMSolution.multiply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsValidatedZeroOutput() {
        let report = P005GEMMJudge.evaluate { lhs, rhs in
            guard lhs.rank == 2 else {
                throw TensorError.rankMismatch(expected: 2, actual: lhs.rank)
            }
            guard rhs.rank == 2 else {
                throw TensorError.rankMismatch(expected: 2, actual: rhs.rank)
            }
            guard lhs.shape[1] == rhs.shape[0] else {
                throw DenseLinearAlgebraError.innerDimensionMismatch(
                    operation: "GEMM",
                    lhs: lhs.shape[1],
                    rhs: rhs.shape[0]
                )
            }
            let shape = [lhs.shape[0], rhs.shape[1]]
            return try FloatTensor(Array(repeating: 0, count: shape[0] * shape[1]), shape: shape)
        }
        XCTAssertFalse(report.isPassing)
    }
}