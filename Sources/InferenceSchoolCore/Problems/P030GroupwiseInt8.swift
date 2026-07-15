import Foundation

public struct GroupwiseInt8Comparison: Sendable, Equatable {
  public let perTensor: SymmetricInt8Tensor
  public let perTensorDequantized: FloatTensor
  public let groupwise: GroupwiseInt8WeightMatrix
  public let groupwiseDequantized: FloatTensor
  public let perTensorError: QuantizationErrorMetrics
  public let groupwiseError: QuantizationErrorMetrics

  public init(
    perTensor: SymmetricInt8Tensor,
    perTensorDequantized: FloatTensor,
    groupwise: GroupwiseInt8WeightMatrix,
    groupwiseDequantized: FloatTensor,
    perTensorError: QuantizationErrorMetrics,
    groupwiseError: QuantizationErrorMetrics
  ) {
    self.perTensor = perTensor
    self.perTensorDequantized = perTensorDequantized
    self.groupwise = groupwise
    self.groupwiseDequantized = groupwiseDequantized
    self.perTensorError = perTensorError
    self.groupwiseError = groupwiseError
  }
}

public typealias GroupwiseInt8Implementation = (
  _ weights: FloatTensor,
  _ groupSize: Int
) throws -> GroupwiseInt8Comparison

public enum P030GroupwiseInt8Judge {
  public static func evaluate(_ implementation: GroupwiseInt8Implementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let weights = try FloatTensor([
        -1, -0.5, 0, 10, 5,
        0.1, -0.1, 0.05, -0.02, 0,
      ], shape: [2, 5])
      let actual = try implementation(weights, 3)
      let expectedScales: [Float] = [1 / 127, 10 / 127, 0.1 / 127, 0.02 / 127]
      if actual.groupwise.shape == [2, 5], actual.groupwise.groupSize == 3,
        actual.groupwise.groupsPerOutputChannel == 2,
        approximatelyEqual(actual.groupwise.scales, expectedScales),
        actual.groupwise.scales.count == 4,
        actual.groupwise.allocatedBytes == 26
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "row groups and tail metadata",
          message: "expected [out=2, groups=2] scales, a two-value tail group, and 26 bytes"))
      }
      let expected = dequantize(actual.groupwise)
      if actual.groupwiseDequantized.shape == weights.shape,
        approximatelyEqual(actual.groupwiseDequantized.storage, expected)
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "groupwise dequantization",
          message: "values did not use scale[row, input/groupSize]"))
      }
      let groupError = metrics(weights.storage, expected)
      let tensorExpected = actual.perTensor.values.map { Float($0) * actual.perTensor.scale }
      let tensorError = metrics(weights.storage, tensorExpected)
      if actual.perTensorDequantized.shape == weights.shape,
        approximatelyEqual(actual.perTensorDequantized.storage, tensorExpected),
        close(actual.groupwiseError, groupError), close(actual.perTensorError, tensorError),
        actual.groupwiseError.rootMeanSquareError < actual.perTensorError.rootMeanSquareError
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "error comparison against per-tensor scale",
          message: "reported metrics must match reconstruction; this fixture favors smaller groups"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "groupwise fixture", message: error.localizedDescription))
    }

    do {
      _ = try implementation(FloatTensor([1, 2], shape: [2]), 1)
      failures.append(JudgeFailure(caseName: "reject rank", message: "expected rank-2 weights"))
    } catch { passed += 1 }
    do {
      _ = try implementation(FloatTensor([1, 2], shape: [1, 2]), 0)
      failures.append(JudgeFailure(
        caseName: "reject group size", message: "expected nonpositive group size to throw"))
    } catch { passed += 1 }
    do {
      _ = try implementation(FloatTensor([1, .nan], shape: [1, 2]), 2)
      failures.append(JudgeFailure(
        caseName: "reject non-finite weight", message: "expected NaN weight to throw"))
    } catch { passed += 1 }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 6, failures: failures)
  }

  private static func dequantize(_ matrix: GroupwiseInt8WeightMatrix) -> [Float] {
    (0..<matrix.outputChannels).flatMap { row in
      (0..<matrix.inputChannels).map { column in
        Float(matrix.values[row * matrix.inputChannels + column])
          * matrix.scales[matrix.scaleIndex(outputChannel: row, inputChannel: column)]
      }
    }
  }

  private static func metrics(_ reference: [Float], _ candidate: [Float]) -> QuantizationErrorMetrics {
    var maximum = 0.0
    var sumSquares = 0.0
    for (expected, actual) in zip(reference, candidate) {
      let difference = Double(expected) - Double(actual)
      maximum = max(maximum, abs(difference))
      sumSquares += difference * difference
    }
    return QuantizationErrorMetrics(
      maximumAbsoluteError: Float(maximum),
      rootMeanSquareError: reference.isEmpty ? 0 : Float(sqrt(sumSquares / Double(reference.count))))
  }

  private static func close(
    _ lhs: QuantizationErrorMetrics,
    _ rhs: QuantizationErrorMetrics
  ) -> Bool {
    abs(lhs.maximumAbsoluteError - rhs.maximumAbsoluteError) <= 1e-6
      && abs(lhs.rootMeanSquareError - rhs.rootMeanSquareError) <= 1e-6
  }

  private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
    lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { abs($0 - $1) <= 1e-6 }
  }
}