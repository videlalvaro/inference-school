import Darwin
import InferenceSchoolCore

public enum P034QuantizationPropagationSolution {
  public static func investigate(
    _ request: QuantizationPropagationRequest
  ) throws -> QuantizationPropagationReport {
    var floatState = request.initial.storage
    var int8State = request.initial.storage
    var q4State = request.initial.storage
    var mismatchState = request.initial.storage
    var captures: [QuantizationLayerCapture] = []
    var changedDecodedValues = 0

    for (layer, floatWeights) in request.weights.enumerated() {
      let int8Weights = try WeightQuantizationSolutionSupport.quantizeGroupwiseInt8(
        floatWeights, groupSize: request.groupSize)
      let q4Weights = try WeightQuantizationSolutionSupport.quantizeGroupwiseQ4(
        floatWeights, groupSize: request.groupSize)
      changedDecodedValues += changedValueCount(q4Weights)

      floatState = activate(floatGEMV(floatWeights.storage, input: floatState, width: floatWeights.shape[0]))
      int8State = activate(int8GEMV(int8Weights, input: int8State))
      q4State = activate(q4GEMV(q4Weights, input: q4State, reverseNibbleOrder: false))
      mismatchState = activate(q4GEMV(
        q4Weights, input: mismatchState, reverseNibbleOrder: true))

      let floatOutput = try FloatTensor(floatState, shape: [floatWeights.shape[0]])
      let int8Output = try FloatTensor(int8State, shape: [floatWeights.shape[0]])
      let q4Output = try FloatTensor(q4State, shape: [floatWeights.shape[0]])
      let mismatchOutput = try FloatTensor(mismatchState, shape: [floatWeights.shape[0]])
      captures.append(QuantizationLayerCapture(
        layer: layer,
        float32Output: floatOutput,
        int8Output: int8Output,
        q4Output: q4Output,
        mismatchedQ4Output: mismatchOutput,
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
        changedDecodedValueCount: changedDecodedValues,
        firstDivergentLayer: captures.first { !$0.mismatchMetrics.argmaxAgreement }?.layer,
        classification: changedDecodedValues > 0 ? .conventionMismatch : .inconclusive))
  }

  private static func floatGEMV(_ weights: [Float], input: [Float], width: Int) -> [Float] {
    (0..<width).map { row in
      var sum: Float = 0
      for column in 0..<width {
        sum += weights[row * width + column] * input[column]
      }
      return sum
    }
  }

  private static func int8GEMV(
    _ weights: GroupwiseInt8WeightMatrix,
    input: [Float]
  ) -> [Float] {
    (0..<weights.outputChannels).map { row in
      var sum: Float = 0
      for column in 0..<weights.inputChannels {
        let index = row * weights.inputChannels + column
        sum += Float(weights.values[index])
          * weights.scales[weights.scaleIndex(outputChannel: row, inputChannel: column)]
          * input[column]
      }
      return sum
    }
  }

  private static func q4GEMV(
    _ weights: GroupwiseQ4WeightMatrix,
    input: [Float],
    reverseNibbleOrder: Bool
  ) -> [Float] {
    (0..<weights.outputChannels).map { row in
      var sum: Float = 0
      for column in 0..<weights.inputChannels {
        let logicalIndex = row * weights.inputChannels + column
        let quantized = reverseNibbleOrder
          ? reversedNibbleValue(weights, logicalIndex: logicalIndex)
          : weights.quantizedValue(outputChannel: row, inputChannel: column)
        sum += Float(quantized)
          * weights.scales[weights.scaleIndex(outputChannel: row, inputChannel: column)]
          * input[column]
      }
      return sum
    }
  }

  private static func reversedNibbleValue(
    _ weights: GroupwiseQ4WeightMatrix,
    logicalIndex: Int
  ) -> Int8 {
    let byte = weights.packedValues[logicalIndex / 2]
    let nibble = logicalIndex.isMultiple(of: 2) ? byte >> 4 : byte & 0x0f
    return Int8(bitPattern: nibble >= 8 ? nibble | 0xf0 : nibble)
  }

  private static func changedValueCount(_ weights: GroupwiseQ4WeightMatrix) -> Int {
    (0..<weights.logicalValueCount).reduce(0) { count, logicalIndex in
      let row = logicalIndex / weights.inputChannels
      let column = logicalIndex % weights.inputChannels
      return count + (weights.quantizedValue(outputChannel: row, inputChannel: column)
        == reversedNibbleValue(weights, logicalIndex: logicalIndex) ? 0 : 1)
    }
  }

  private static func activate(_ values: [Float]) -> [Float] {
    values.map { tanh($0) }
  }

  private static func metrics(
    reference: [Float], candidate: [Float]
  ) -> VectorComparisonMetrics {
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
}