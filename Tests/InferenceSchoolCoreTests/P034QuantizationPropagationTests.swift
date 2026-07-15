import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P034QuantizationPropagationTests: XCTestCase {
  func testCanonicalInvestigationPassesJudge() {
    let report = P034QuantizationPropagationJudge.evaluate(
      P034QuantizationPropagationSolution.investigate)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testKnownConventionInterventionProducesEvidence() throws {
    let request = try QuantizationPropagationRequest(
      initial: FloatTensor([1, -0.5, 0.25], shape: [3]),
      weights: [FloatTensor([
        1, 0.5, -1,
        -0.25, 2, 0.75,
        0.1, -0.2, 0.3,
      ], shape: [3, 3])],
      groupSize: 2)
    let report = try P034QuantizationPropagationSolution.investigate(request)
    XCTAssertEqual(report.mismatchDiagnostic.classification, .conventionMismatch)
    XCTAssertGreaterThan(report.mismatchDiagnostic.changedDecodedValueCount, 0)
    XCTAssertEqual(report.layers.count, 1)
  }

  func testJudgeRejectsThresholdOnlyInconclusiveClassification() {
    let report = P034QuantizationPropagationJudge.evaluate { request in
      let actual = try P034QuantizationPropagationSolution.investigate(request)
      return QuantizationPropagationReport(
        layers: actual.layers,
        firstInt8DivergentLayer: actual.firstInt8DivergentLayer,
        firstQ4DivergentLayer: actual.firstQ4DivergentLayer,
        mismatchDiagnostic: ConventionMismatchDiagnostic(
          injectedFault: .highNibbleFirstInsteadOfLowNibbleFirst,
          changedDecodedValueCount: actual.mismatchDiagnostic.changedDecodedValueCount,
          firstDivergentLayer: actual.mismatchDiagnostic.firstDivergentLayer,
          classification: .inconclusive))
    }
    XCTAssertFalse(report.isPassing)
  }
}