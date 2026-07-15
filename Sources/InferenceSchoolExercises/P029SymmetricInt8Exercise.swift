import Foundation
import InferenceSchoolCore

public enum P029SymmetricInt8Exercise {
  public static func quantize(
    _ input: FloatTensor
  ) throws -> SymmetricInt8QuantizationResult {
    for (index, value) in input.storage.enumerated() where !value.isFinite {
      throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
    }
    let quantized = try SymmetricInt8Tensor(
      values: Array(repeating: 0, count: input.elementCount),
      shape: input.shape,
      scale: 1)
    let dequantized = try FloatTensor(
      Array(repeating: 0, count: input.elementCount), shape: input.shape)
    let maximum = input.storage.map(abs).max() ?? 0
    let sumSquares = input.storage.reduce(0.0) { $0 + Double($1) * Double($1) }
    return SymmetricInt8QuantizationResult(
      quantized: quantized,
      dequantized: dequantized,
      error: QuantizationErrorMetrics(
        maximumAbsoluteError: maximum,
        rootMeanSquareError: input.storage.isEmpty
          ? 0 : Float(sqrt(sumSquares / Double(input.elementCount)))))
  }
}