import InferenceSchoolCore

public enum P033FusedQ4GEMVSolution {
  public static func quantize(
    _ weights: FloatTensor,
    groupSize: Int
  ) throws -> GroupwiseQ4WeightMatrix {
    try WeightQuantizationSolutionSupport.quantizeGroupwiseQ4(
      weights, groupSize: groupSize)
  }

  public static func multiply(
    _ weights: GroupwiseQ4WeightMatrix,
    _ input: FloatTensor
  ) throws -> FusedQ4GEMVResult {
    try validate(weights: weights, input: input)
    var output = Array(repeating: Float.zero, count: weights.outputChannels)
    for row in 0..<weights.outputChannels {
      var sum: Float = 0
      for column in 0..<weights.inputChannels {
        let quantized = weights.quantizedValue(outputChannel: row, inputChannel: column)
        let scale = weights.scales[weights.scaleIndex(
          outputChannel: row, inputChannel: column)]
        sum += Float(quantized) * scale * input.storage[column]
      }
      output[row] = sum
    }
    return FusedQ4GEMVResult(
      output: try FloatTensor(output, shape: [weights.outputChannels]),
      logicalWeightBytes: weights.allocatedBytes,
      temporaryWeightBytes: 0)
  }

  static func validate(
    weights: GroupwiseQ4WeightMatrix,
    input: FloatTensor
  ) throws {
    guard input.rank == 1 else {
      throw TensorError.rankMismatch(expected: 1, actual: input.rank)
    }
    guard input.shape[0] == weights.inputChannels else {
      throw DenseLinearAlgebraError.innerDimensionMismatch(
        operation: "fused Q4 GEMV",
        lhs: weights.inputChannels,
        rhs: input.shape[0])
    }
    try WeightQuantizationSolutionSupport.validateFinite(input.storage)
  }
}