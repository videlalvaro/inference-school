import InferenceSchoolCore

public enum P030GroupwiseInt8Exercise {
  public static func compare(
    _ weights: FloatTensor,
    groupSize: Int
  ) throws -> GroupwiseInt8Comparison {
    guard weights.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: weights.rank)
    }
    guard groupSize > 0 else { throw WeightQuantizationError.invalidGroupSize(groupSize) }
    for (index, value) in weights.storage.enumerated() where !value.isFinite {
      throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
    }
    let outputChannels = weights.shape[0]
    let inputChannels = weights.shape[1]
    let groups = inputChannels == 0 ? 0 : (inputChannels + groupSize - 1) / groupSize
    let perTensor = try SymmetricInt8Tensor(
      values: Array(repeating: 0, count: weights.elementCount),
      shape: weights.shape,
      scale: 1)
    let groupwise = try GroupwiseInt8WeightMatrix(
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      groupSize: groupSize,
      values: Array(repeating: 0, count: weights.elementCount),
      scales: Array(repeating: 1, count: outputChannels * groups))
    let zeros = try FloatTensor(
      Array(repeating: 0, count: weights.elementCount), shape: weights.shape)
    let error = QuantizationErrorMetrics(maximumAbsoluteError: 0, rootMeanSquareError: 0)
    return GroupwiseInt8Comparison(
      perTensor: perTensor,
      perTensorDequantized: zeros,
      groupwise: groupwise,
      groupwiseDequantized: zeros,
      perTensorError: error,
      groupwiseError: error)
  }
}