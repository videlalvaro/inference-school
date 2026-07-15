import InferenceSchoolCore

public enum P033FusedQ4GEMVExercise {
  public static func multiply(
    _ weights: GroupwiseQ4WeightMatrix,
    _ input: FloatTensor
  ) throws -> FusedQ4GEMVResult {
    guard input.rank == 1 else {
      throw TensorError.rankMismatch(expected: 1, actual: input.rank)
    }
    guard input.shape[0] == weights.inputChannels else {
      throw DenseLinearAlgebraError.innerDimensionMismatch(
        operation: "fused Q4 GEMV", lhs: weights.inputChannels, rhs: input.shape[0])
    }
    for (index, value) in input.storage.enumerated() where !value.isFinite {
      throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
    }
    return FusedQ4GEMVResult(
      output: try FloatTensor(
        Array(repeating: 0, count: weights.outputChannels), shape: [weights.outputChannels]),
      logicalWeightBytes: weights.allocatedBytes,
      temporaryWeightBytes: 0)
  }
}