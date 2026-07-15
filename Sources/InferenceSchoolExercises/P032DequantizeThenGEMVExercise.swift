import InferenceSchoolCore

public enum P032DequantizeThenGEMVExercise {
  public static func multiply(
    _ weights: GroupwiseQ4WeightMatrix,
    _ input: FloatTensor
  ) throws -> DequantizeThenGEMVResult {
    guard input.rank == 1 else {
      throw TensorError.rankMismatch(expected: 1, actual: input.rank)
    }
    guard input.shape[0] == weights.inputChannels else {
      throw DenseLinearAlgebraError.innerDimensionMismatch(
        operation: "dequantize-then-GEMV", lhs: weights.inputChannels, rhs: input.shape[0])
    }
    for (index, value) in input.storage.enumerated() where !value.isFinite {
      throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
    }
    let materialized = try FloatTensor(
      Array(repeating: 0, count: weights.logicalValueCount), shape: weights.shape)
    return DequantizeThenGEMVResult(
      output: try FloatTensor(
        Array(repeating: 0, count: weights.outputChannels), shape: [weights.outputChannels]),
      materializedWeights: materialized,
      temporaryWeightBytes: weights.logicalValueCount * MemoryLayout<Float>.stride)
  }
}