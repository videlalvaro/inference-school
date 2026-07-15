import InferenceSchoolCore

public enum P032DequantizeThenGEMVSolution {
  public static func multiply(
    _ weights: GroupwiseQ4WeightMatrix,
    _ input: FloatTensor
  ) throws -> DequantizeThenGEMVResult {
    guard input.rank == 1 else {
      throw TensorError.rankMismatch(expected: 1, actual: input.rank)
    }
    guard input.shape[0] == weights.inputChannels else {
      throw DenseLinearAlgebraError.innerDimensionMismatch(
        operation: "dequantize-then-GEMV",
        lhs: weights.inputChannels,
        rhs: input.shape[0])
    }
    try WeightQuantizationSolutionSupport.validateFinite(input.storage)
    let materialized = try WeightQuantizationSolutionSupport.dequantize(weights)
    return DequantizeThenGEMVResult(
      output: try P004GEMVSolution.multiply(matrix: materialized, vector: input),
      materializedWeights: materialized,
      temporaryWeightBytes: weights.logicalValueCount * MemoryLayout<Float>.stride)
  }
}