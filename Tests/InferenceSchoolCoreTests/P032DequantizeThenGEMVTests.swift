import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P032DequantizeThenGEMVTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P032DequantizeThenGEMVJudge.evaluate(
      P032DequantizeThenGEMVSolution.multiply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsUnreportedMaterialization() {
    let report = P032DequantizeThenGEMVJudge.evaluate { weights, input in
      let actual = try P032DequantizeThenGEMVSolution.multiply(weights, input)
      return DequantizeThenGEMVResult(
        output: actual.output,
        materializedWeights: actual.materializedWeights,
        temporaryWeightBytes: 0)
    }
    XCTAssertFalse(report.isPassing)
  }
}