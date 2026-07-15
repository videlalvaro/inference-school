import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P029SymmetricInt8Tests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P029SymmetricInt8Judge.evaluate(P029SymmetricInt8Solution.quantize)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsTruncationAndMissingScale() {
    let report = P029SymmetricInt8Judge.evaluate { input in
      for (index, value) in input.storage.enumerated() where !value.isFinite {
        throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
      }
      let quantized = try SymmetricInt8Tensor(
        values: input.storage.map { Int8(max(-127, min(127, Int($0)))) },
        shape: input.shape,
        scale: 1)
      return SymmetricInt8QuantizationResult(
        quantized: quantized,
        dequantized: try FloatTensor(quantized.values.map(Float.init), shape: input.shape),
        error: QuantizationErrorMetrics(maximumAbsoluteError: 0, rootMeanSquareError: 0))
    }
    XCTAssertFalse(report.isPassing)
  }

  func testFormatRejectsAsymmetricMinus128() {
    XCTAssertThrowsError(try SymmetricInt8Tensor(values: [-128], shape: [1], scale: 1))
  }
}