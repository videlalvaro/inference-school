import Foundation

public struct SymmetricInt8QuantizationResult: Sendable, Equatable {
  public let quantized: SymmetricInt8Tensor
  public let dequantized: FloatTensor
  public let error: QuantizationErrorMetrics

  public init(
    quantized: SymmetricInt8Tensor,
    dequantized: FloatTensor,
    error: QuantizationErrorMetrics
  ) {
    self.quantized = quantized
    self.dequantized = dequantized
    self.error = error
  }
}

public typealias SymmetricInt8QuantizationImplementation = (
  _ input: FloatTensor
) throws -> SymmetricInt8QuantizationResult

public enum P029SymmetricInt8Judge {
  public static func evaluate(
    _ implementation: SymmetricInt8QuantizationImplementation
  ) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0

    do {
      let input = try FloatTensor([-2, -1, 0, 1, 2], shape: [5])
      let actual = try implementation(input)
      let expectedValues: [Int8] = [-127, -64, 0, 64, 127]
      let expectedScale: Float = 2 / 127
      let expectedDequantized = expectedValues.map { Float($0) * expectedScale }
      if actual.quantized.shape == input.shape,
        actual.quantized.values == expectedValues,
        abs(actual.quantized.scale - expectedScale) <= 1e-7,
        approximatelyEqual(actual.dequantized.storage, expectedDequantized),
        actual.quantized.allocatedBytes == 9
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "signed range, saturation, and tie rounding",
          message: "expected [-127,-64,0,64,127], scale 2/127, shape preservation, and 9 bytes"))
      }
      let oracleError = metrics(reference: input.storage, candidate: expectedDequantized)
      if abs(actual.error.maximumAbsoluteError - oracleError.maximumAbsoluteError) <= 1e-7,
        abs(actual.error.rootMeanSquareError - oracleError.rootMeanSquareError) <= 1e-7
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "independent error metrics",
          message: "maximum absolute error or RMSE did not match the Double oracle"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "nonzero tensor", message: error.localizedDescription))
    }

    do {
      let zeros = try FloatTensor([0, 0, 0, 0], shape: [2, 2])
      let actual = try implementation(zeros)
      if actual.quantized.values == [0, 0, 0, 0], actual.quantized.scale == 1,
        actual.dequantized == zeros, actual.error == QuantizationErrorMetrics(
          maximumAbsoluteError: 0, rootMeanSquareError: 0)
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "all-zero tensor",
          message: "all-zero input must use finite scale 1 and round-trip exactly"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "all-zero tensor", message: error.localizedDescription))
    }

    do {
      _ = try implementation(FloatTensor([1, .infinity], shape: [2]))
      failures.append(JudgeFailure(
        caseName: "reject non-finite input",
        message: "expected non-finite input to throw"))
    } catch {
      passed += 1
    }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }

  private static func metrics(reference: [Float], candidate: [Float]) -> QuantizationErrorMetrics {
    var maximum = 0.0
    var sumSquares = 0.0
    for (expected, actual) in zip(reference, candidate) {
      let difference = Double(expected) - Double(actual)
      maximum = max(maximum, abs(difference))
      sumSquares += difference * difference
    }
    let rmse = reference.isEmpty ? 0 : sqrt(sumSquares / Double(reference.count))
    return QuantizationErrorMetrics(
      maximumAbsoluteError: Float(maximum), rootMeanSquareError: Float(rmse))
  }

  private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
    lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { abs($0 - $1) <= 1e-6 }
  }
}