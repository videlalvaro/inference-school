import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P008SwiGLUTests: XCTestCase {
    func testCanonicalCPUSolutionPassesJudge() {
        let report = P008SwiGLUJudge.evaluate(P008SwiGLUSolution.apply)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testCanonicalGatePassesJudge() {
        let report = P008SwiGLUJudge.evaluateGate(P008SwiGLUSolution.gate)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsValidatedZeroOutput() {
        let report = P008SwiGLUJudge.evaluate { input, gate, up, down in
            guard input.rank == 1, gate.rank == 2, up.rank == 2, down.rank == 2 else {
                throw TensorError.rankMismatch(expected: 1, actual: input.rank)
            }
            guard gate.shape == up.shape else { throw SwiGLUError.hiddenProjectionShapeMismatch(gate: gate.shape, up: up.shape) }
            guard gate.shape[1] == input.shape[0] else { throw SwiGLUError.inputWidthMismatch(expected: gate.shape[1], actual: input.shape[0]) }
            guard down.shape[1] == gate.shape[0] else { throw SwiGLUError.downWidthMismatch(expected: gate.shape[0], actual: down.shape[1]) }
            return try FloatTensor(Array(repeating: 0, count: down.shape[0]), shape: [down.shape[0]])
        }
        XCTAssertFalse(report.isPassing)
    }
}