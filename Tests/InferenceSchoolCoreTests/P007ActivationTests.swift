import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P007ActivationTests: XCTestCase {
    func testCanonicalCPUSolutionPassesJudge() {
        let report = P007ActivationJudge.evaluate(P007ActivationSolution.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsIdentityImplementation() {
        let report = P007ActivationJudge.evaluate { input, _ in input }
        XCTAssertFalse(report.isPassing)
    }
}