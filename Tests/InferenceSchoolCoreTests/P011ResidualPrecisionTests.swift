import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P011ResidualPrecisionTests: XCTestCase {
    func testCanonicalSolutionPassesJudge() {
        let report = P011ResidualPrecisionJudge.evaluate(P011ResidualPrecisionSolution.accumulate)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testComparisonMeasuresDowncastPlacementError() throws {
        let initial = try FloatTensor([4096], shape: [1])
        let updates = try (0..<4).map { _ in try FloatTensor([0.5], shape: [1]) }
        let comparison = try P011ResidualPrecisionSolution.compare(initial, updates: updates)
        XCTAssertEqual(comparison.float32.storage, [4098])
        XCTAssertEqual(comparison.float16AfterEachAdd.storage, [4096])
        XCTAssertEqual(comparison.maximumAbsoluteDifference, 2)
    }

    func testJudgeRejectsIgnoringUpdates() {
        let report = P011ResidualPrecisionJudge.evaluate { initial, updates, _ in
            for (index, update) in updates.enumerated() where update.shape != initial.shape {
                throw ResidualStreamError.updateShapeMismatch(index: index, expected: initial.shape, actual: update.shape)
            }
            return initial
        }
        XCTAssertFalse(report.isPassing)
    }
}