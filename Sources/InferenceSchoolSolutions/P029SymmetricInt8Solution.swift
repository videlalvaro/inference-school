import InferenceSchoolCore

public enum P029SymmetricInt8Solution {
  public static func quantize(
    _ input: FloatTensor
  ) throws -> SymmetricInt8QuantizationResult {
    let quantized = try WeightQuantizationSolutionSupport.quantizeInt8(input)
    let dequantized = try WeightQuantizationSolutionSupport.dequantize(quantized)
    return SymmetricInt8QuantizationResult(
      quantized: quantized,
      dequantized: dequantized,
      error: WeightQuantizationSolutionSupport.metrics(
        reference: input.storage, candidate: dequantized.storage))
  }
}