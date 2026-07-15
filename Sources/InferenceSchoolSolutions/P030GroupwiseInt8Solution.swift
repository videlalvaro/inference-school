import InferenceSchoolCore

public enum P030GroupwiseInt8Solution {
  public static func compare(
    _ weights: FloatTensor,
    groupSize: Int
  ) throws -> GroupwiseInt8Comparison {
    let perTensor = try WeightQuantizationSolutionSupport.quantizeInt8(weights)
    let perTensorDequantized = try WeightQuantizationSolutionSupport.dequantize(perTensor)
    let groupwise = try WeightQuantizationSolutionSupport.quantizeGroupwiseInt8(
      weights, groupSize: groupSize)
    let groupwiseDequantized = try WeightQuantizationSolutionSupport.dequantize(groupwise)
    return GroupwiseInt8Comparison(
      perTensor: perTensor,
      perTensorDequantized: perTensorDequantized,
      groupwise: groupwise,
      groupwiseDequantized: groupwiseDequantized,
      perTensorError: WeightQuantizationSolutionSupport.metrics(
        reference: weights.storage, candidate: perTensorDequantized.storage),
      groupwiseError: WeightQuantizationSolutionSupport.metrics(
        reference: weights.storage, candidate: groupwiseDequantized.storage))
  }
}