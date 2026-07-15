import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P030GroupwiseInt8Tests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P030GroupwiseInt8Judge.evaluate(P030GroupwiseInt8Solution.compare)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsOneScalePerRowWhenTailNeedsOwnScale() {
    let report = P030GroupwiseInt8Judge.evaluate { weights, groupSize in
      guard weights.rank == 2 else {
        throw TensorError.rankMismatch(expected: 2, actual: weights.rank)
      }
      let valid = try P030GroupwiseInt8Solution.compare(weights, groupSize: weights.shape[1])
      let fake = try GroupwiseInt8WeightMatrix(
        outputChannels: weights.shape[0], inputChannels: weights.shape[1],
        groupSize: groupSize,
        values: valid.groupwise.values,
        scales: Array(repeating: 1, count: weights.shape[0] * 2))
      return GroupwiseInt8Comparison(
        perTensor: valid.perTensor, perTensorDequantized: valid.perTensorDequantized,
        groupwise: fake, groupwiseDequantized: weights,
        perTensorError: valid.perTensorError, groupwiseError: valid.groupwiseError)
    }
    XCTAssertFalse(report.isPassing)
  }
}