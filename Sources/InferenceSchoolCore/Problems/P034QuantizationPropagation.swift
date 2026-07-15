import Foundation

public struct VectorComparisonMetrics: Sendable, Equatable {
  public let cosineSimilarity: Float
  public let maximumAbsoluteError: Float
  public let rootMeanSquareError: Float
  public let argmaxAgreement: Bool

  public init(
    cosineSimilarity: Float,
    maximumAbsoluteError: Float,
    rootMeanSquareError: Float,
    argmaxAgreement: Bool
  ) {
    self.cosineSimilarity = cosineSimilarity
    self.maximumAbsoluteError = maximumAbsoluteError
    self.rootMeanSquareError = rootMeanSquareError
    self.argmaxAgreement = argmaxAgreement
  }
}

public struct QuantizationLayerCapture: Sendable, Equatable {
  public let layer: Int
  public let float32Output: FloatTensor
  public let int8Output: FloatTensor
  public let q4Output: FloatTensor
  public let mismatchedQ4Output: FloatTensor
  public let int8Metrics: VectorComparisonMetrics
  public let q4Metrics: VectorComparisonMetrics
  public let mismatchMetrics: VectorComparisonMetrics

  public init(
    layer: Int,
    float32Output: FloatTensor,
    int8Output: FloatTensor,
    q4Output: FloatTensor,
    mismatchedQ4Output: FloatTensor,
    int8Metrics: VectorComparisonMetrics,
    q4Metrics: VectorComparisonMetrics,
    mismatchMetrics: VectorComparisonMetrics
  ) {
    self.layer = layer
    self.float32Output = float32Output
    self.int8Output = int8Output
    self.q4Output = q4Output
    self.mismatchedQ4Output = mismatchedQ4Output
    self.int8Metrics = int8Metrics
    self.q4Metrics = q4Metrics
    self.mismatchMetrics = mismatchMetrics
  }
}

public enum QuantizationConventionFault: Sendable, Equatable {
  case highNibbleFirstInsteadOfLowNibbleFirst
}

public enum QuantizationDiagnosticClassification: Sendable, Equatable {
  case conventionMismatch
  case inconclusive
}

public struct ConventionMismatchDiagnostic: Sendable, Equatable {
  public let injectedFault: QuantizationConventionFault
  public let changedDecodedValueCount: Int
  public let firstDivergentLayer: Int?
  public let classification: QuantizationDiagnosticClassification

  public init(
    injectedFault: QuantizationConventionFault,
    changedDecodedValueCount: Int,
    firstDivergentLayer: Int?,
    classification: QuantizationDiagnosticClassification
  ) {
    self.injectedFault = injectedFault
    self.changedDecodedValueCount = changedDecodedValueCount
    self.firstDivergentLayer = firstDivergentLayer
    self.classification = classification
  }
}

public struct QuantizationPropagationReport: Sendable, Equatable {
  public let layers: [QuantizationLayerCapture]
  public let firstInt8DivergentLayer: Int?
  public let firstQ4DivergentLayer: Int?
  public let mismatchDiagnostic: ConventionMismatchDiagnostic

  public init(
    layers: [QuantizationLayerCapture],
    firstInt8DivergentLayer: Int?,
    firstQ4DivergentLayer: Int?,
    mismatchDiagnostic: ConventionMismatchDiagnostic
  ) {
    self.layers = layers
    self.firstInt8DivergentLayer = firstInt8DivergentLayer
    self.firstQ4DivergentLayer = firstQ4DivergentLayer
    self.mismatchDiagnostic = mismatchDiagnostic
  }
}

public struct QuantizationPropagationRequest: Sendable, Equatable {
  public let initial: FloatTensor
  public let weights: [FloatTensor]
  public let groupSize: Int

  public init(initial: FloatTensor, weights: [FloatTensor], groupSize: Int) throws {
    guard initial.rank == 1 else {
      throw TensorError.rankMismatch(expected: 1, actual: initial.rank)
    }
    guard groupSize > 0 else { throw WeightQuantizationError.invalidGroupSize(groupSize) }
    for (index, value) in initial.storage.enumerated() where !value.isFinite {
      throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
    }
    let width = initial.shape[0]
    for (layer, weight) in weights.enumerated() {
      guard weight.shape == [width, width] else {
        throw WeightQuantizationError.shapeMismatch(
          name: "Layer \(layer) weights", expected: [width, width], actual: weight.shape)
      }
      for (index, value) in weight.storage.enumerated() where !value.isFinite {
        throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
      }
    }
    self.initial = initial
    self.weights = weights
    self.groupSize = groupSize
  }
}

public typealias QuantizationPropagationImplementation = (
  _ request: QuantizationPropagationRequest
) throws -> QuantizationPropagationReport

public enum P034QuantizationPropagationJudge {
  public static func evaluate(
    _ implementation: QuantizationPropagationImplementation
  ) -> JudgeReport {
    let request: QuantizationPropagationRequest
    do {
      request = try fixture()
    } catch {
      return JudgeReport(
        passedCaseCount: 0, totalCaseCount: 4,
        failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)])
    }
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let actual = try implementation(request)
      let expected = try oracle(request)
      if actual.layers.count == expected.layers.count,
        zip(actual.layers, expected.layers).allSatisfy(capturesApproximatelyEqual)
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "deterministic per-layer captures",
          message: "FP32, INT8, Q4, or mismatched outputs did not match the independent pipeline"))
      }
      if zip(actual.layers, expected.layers).allSatisfy({
        metricsApproximatelyEqual($0.int8Metrics, $1.int8Metrics)
          && metricsApproximatelyEqual($0.q4Metrics, $1.q4Metrics)
          && metricsApproximatelyEqual($0.mismatchMetrics, $1.mismatchMetrics)
      }) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "cosine, maximum error, RMSE, and argmax",
          message: "one or more reported metrics did not match captured outputs"))
      }
      if actual.firstInt8DivergentLayer == expected.firstInt8DivergentLayer,
        actual.firstQ4DivergentLayer == expected.firstQ4DivergentLayer
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "first divergent layer",
          message: "divergence must be derived from the first argmax disagreement"))
      }
      if actual.mismatchDiagnostic == expected.mismatchDiagnostic,
        actual.mismatchDiagnostic.changedDecodedValueCount > 0,
        actual.mismatchDiagnostic.classification == .conventionMismatch
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "controlled convention mismatch diagnosis",
          message: "classification requires a known nibble-order intervention and changed decodes"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "propagation report", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }

  private static func fixture() throws -> QuantizationPropagationRequest {
    let width = 5
    let weights = (0..<3).map { layer in
      (0..<(width * width)).map { index -> Float in
        let signed = ((index * 7 + layer * 5) % 19) - 9
        let rowScale: Float = index / width == layer ? 1.7 : 0.35
        return Float(signed) * rowScale / 5
      }
    }
    return try QuantizationPropagationRequest(
      initial: FloatTensor([0.75, -1.25, 0.5, 1.5, -0.25], shape: [width]),
      weights: try weights.map { try FloatTensor($0, shape: [width, width]) },
      groupSize: 3)
  }

  private static func oracle(
    _ request: QuantizationPropagationRequest
  ) throws -> QuantizationPropagationReport {
    var floatState = request.initial.storage
    var int8State = request.initial.storage
    var q4State = request.initial.storage
    var mismatchState = request.initial.storage
    var captures: [QuantizationLayerCapture] = []
    var changedDecodedValueCount = 0
    for (layer, weights) in request.weights.enumerated() {
      let int8 = quantize(weights, groupSize: request.groupSize, limit: 127)
      let q4 = quantize(weights, groupSize: request.groupSize, limit: 7)
      changedDecodedValueCount += zip(q4.values, swappedPairs(q4.values)).filter(!=).count
      floatState = activatedGEMV(weights.storage, input: floatState, width: weights.shape[0])
      int8State = activatedGEMV(
        dequantized(int8), input: int8State, width: weights.shape[0])
      q4State = activatedGEMV(dequantized(q4), input: q4State, width: weights.shape[0])
      mismatchState = activatedGEMV(
        dequantized(q4, values: swappedPairs(q4.values)),
        input: mismatchState,
        width: weights.shape[0])
      let floatTensor = try FloatTensor(floatState, shape: [weights.shape[0]])
      let int8Tensor = try FloatTensor(int8State, shape: [weights.shape[0]])
      let q4Tensor = try FloatTensor(q4State, shape: [weights.shape[0]])
      let mismatchTensor = try FloatTensor(mismatchState, shape: [weights.shape[0]])
      captures.append(QuantizationLayerCapture(
        layer: layer,
        float32Output: floatTensor,
        int8Output: int8Tensor,
        q4Output: q4Tensor,
        mismatchedQ4Output: mismatchTensor,
        int8Metrics: metrics(reference: floatState, candidate: int8State),
        q4Metrics: metrics(reference: floatState, candidate: q4State),
        mismatchMetrics: metrics(reference: floatState, candidate: mismatchState)))
    }
    return QuantizationPropagationReport(
      layers: captures,
      firstInt8DivergentLayer: captures.first { !$0.int8Metrics.argmaxAgreement }?.layer,
      firstQ4DivergentLayer: captures.first { !$0.q4Metrics.argmaxAgreement }?.layer,
      mismatchDiagnostic: ConventionMismatchDiagnostic(
        injectedFault: .highNibbleFirstInsteadOfLowNibbleFirst,
        changedDecodedValueCount: changedDecodedValueCount,
        firstDivergentLayer: captures.first { !$0.mismatchMetrics.argmaxAgreement }?.layer,
        classification: changedDecodedValueCount > 0 ? .conventionMismatch : .inconclusive))
  }

  private struct OracleQuantized {
    let values: [Int]
    let scales: [Float]
    let width: Int
    let groupSize: Int
  }

  private static func quantize(
    _ weights: FloatTensor, groupSize: Int, limit: Int
  ) -> OracleQuantized {
    let width = weights.shape[0]
    let groups = (width + groupSize - 1) / groupSize
    var values = Array(repeating: 0, count: weights.elementCount)
    var scales = Array(repeating: Float.zero, count: width * groups)
    for row in 0..<width {
      for group in 0..<groups {
        let start = group * groupSize
        let end = min(start + groupSize, width)
        let maximum = (start..<end).reduce(Float.zero) {
          max($0, abs(weights.storage[row * width + $1]))
        }
        let scale: Float = maximum == 0 ? 1 : maximum / Float(limit)
        scales[row * groups + group] = scale
        for column in start..<end {
          let rounded = weights.storage[row * width + column]
            .divided(by: scale).rounded(.toNearestOrAwayFromZero)
          values[row * width + column] = max(-limit - (limit == 7 ? 1 : 0), min(limit, Int(rounded)))
        }
      }
    }
    return OracleQuantized(values: values, scales: scales, width: width, groupSize: groupSize)
  }

  private static func dequantized(
    _ quantized: OracleQuantized, values: [Int]? = nil
  ) -> [Float] {
    let source = values ?? quantized.values
    let groups = (quantized.width + quantized.groupSize - 1) / quantized.groupSize
    return source.enumerated().map { index, value in
      let row = index / quantized.width
      let column = index % quantized.width
      return Float(value) * quantized.scales[
        row * groups + column / quantized.groupSize]
    }
  }

  private static func swappedPairs(_ values: [Int]) -> [Int] {
    var result = values
    for index in stride(from: 0, to: values.count - 1, by: 2) {
      result[index] = values[index + 1]
      result[index + 1] = values[index]
    }
    if values.count % 2 == 1 { result[values.count - 1] = 0 }
    return result
  }

  private static func activatedGEMV(_ weights: [Float], input: [Float], width: Int) -> [Float] {
    (0..<width).map { row in
      let sum = (0..<width).reduce(0.0) { partial, column in
        partial + Double(weights[row * width + column]) * Double(input[column])
      }
      return Float(tanh(sum))
    }
  }

  private static func metrics(reference: [Float], candidate: [Float]) -> VectorComparisonMetrics {
    var dot = 0.0
    var referenceNorm = 0.0
    var candidateNorm = 0.0
    var maximum = 0.0
    var sumSquares = 0.0
    for (expected, actual) in zip(reference, candidate) {
      let left = Double(expected)
      let right = Double(actual)
      let difference = left - right
      dot += left * right
      referenceNorm += left * left
      candidateNorm += right * right
      maximum = max(maximum, abs(difference))
      sumSquares += difference * difference
    }
    let denominator = sqrt(referenceNorm * candidateNorm)
    let cosine: Double = denominator == 0
      ? (referenceNorm == candidateNorm ? 1 : 0)
      : dot / denominator
    return VectorComparisonMetrics(
      cosineSimilarity: Float(cosine),
      maximumAbsoluteError: Float(maximum),
      rootMeanSquareError: reference.isEmpty ? 0 : Float(sqrt(sumSquares / Double(reference.count))),
      argmaxAgreement: argmax(reference) == argmax(candidate))
  }

  private static func argmax(_ values: [Float]) -> Int? {
    values.indices.max { values[$0] < values[$1] }
  }

  private static func capturesApproximatelyEqual(
    _ lhs: QuantizationLayerCapture, _ rhs: QuantizationLayerCapture
  ) -> Bool {
    lhs.layer == rhs.layer
      && tensorsApproximatelyEqual(lhs.float32Output, rhs.float32Output)
      && tensorsApproximatelyEqual(lhs.int8Output, rhs.int8Output)
      && tensorsApproximatelyEqual(lhs.q4Output, rhs.q4Output)
      && tensorsApproximatelyEqual(lhs.mismatchedQ4Output, rhs.mismatchedQ4Output)
  }

  private static func tensorsApproximatelyEqual(_ lhs: FloatTensor, _ rhs: FloatTensor) -> Bool {
    lhs.shape == rhs.shape && zip(lhs.storage, rhs.storage).allSatisfy {
      abs($0 - $1) <= 2e-5 * max(1, abs($0), abs($1))
    }
  }

  private static func metricsApproximatelyEqual(
    _ lhs: VectorComparisonMetrics, _ rhs: VectorComparisonMetrics
  ) -> Bool {
    abs(lhs.cosineSimilarity - rhs.cosineSimilarity) <= 2e-5
      && abs(lhs.maximumAbsoluteError - rhs.maximumAbsoluteError) <= 2e-5
      && abs(lhs.rootMeanSquareError - rhs.rootMeanSquareError) <= 2e-5
      && lhs.argmaxAgreement == rhs.argmaxAgreement
  }
}

private extension Float {
  func divided(by divisor: Float) -> Float { self / divisor }
}