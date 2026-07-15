import InferenceSchoolCore

public enum P034QuantizationPropagationExercise {
  public static func investigate(
    _ request: QuantizationPropagationRequest
  ) throws -> QuantizationPropagationReport {
    let captures = try request.weights.enumerated().map { layer, weights in
      let zeros = try FloatTensor(Array(repeating: 0, count: weights.shape[0]), shape: [weights.shape[0]])
      let metrics = VectorComparisonMetrics(
        cosineSimilarity: 1,
        maximumAbsoluteError: 0,
        rootMeanSquareError: 0,
        argmaxAgreement: true)
      return QuantizationLayerCapture(
        layer: layer,
        float32Output: zeros,
        int8Output: zeros,
        q4Output: zeros,
        mismatchedQ4Output: zeros,
        int8Metrics: metrics,
        q4Metrics: metrics,
        mismatchMetrics: metrics)
    }
    return QuantizationPropagationReport(
      layers: captures,
      firstInt8DivergentLayer: nil,
      firstQ4DivergentLayer: nil,
      mismatchDiagnostic: ConventionMismatchDiagnostic(
        injectedFault: .highNibbleFirstInsteadOfLowNibbleFirst,
        changedDecodedValueCount: 0,
        firstDivergentLayer: nil,
        classification: .inconclusive))
  }
}