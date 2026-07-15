import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P010RMSNormTests: XCTestCase {
    func testCanonicalCPUSolutionPassesJudge() {
        let report = P010RMSNormJudge.evaluate(P010RMSNormSolution.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsIdentityImplementation() {
        let report = P010RMSNormJudge.evaluate { input, gamma, epsilon in
            guard input.rank == 2, gamma.rank == 1, epsilon > 0 else {
                throw RMSNormError.invalidEpsilon(epsilon)
            }
            return input
        }
        XCTAssertFalse(report.isPassing)
    }
}