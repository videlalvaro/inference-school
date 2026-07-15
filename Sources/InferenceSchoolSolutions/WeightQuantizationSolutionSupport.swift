import Foundation
import InferenceSchoolCore

enum WeightQuantizationSolutionSupport {
  static func validateFinite(_ values: [Float]) throws {
    for (index, value) in values.enumerated() where !value.isFinite {
      throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
    }
  }

  static func quantizeInt8(_ input: FloatTensor) throws -> SymmetricInt8Tensor {
    try validateFinite(input.storage)
    let maximum = input.storage.reduce(Float.zero) { max($0, abs($1)) }
    let scale: Float = maximum == 0 ? 1 : maximum / 127
    let values = input.storage.map { value -> Int8 in
      let rounded = QuantizationRounding.nearestAwayFromZero(value / scale)
      return Int8(max(-127, min(127, Int(rounded))))
    }
    return try SymmetricInt8Tensor(values: values, shape: input.shape, scale: scale)
  }

  static func dequantize(_ input: SymmetricInt8Tensor) throws -> FloatTensor {
    try FloatTensor(input.values.map { Float($0) * input.scale }, shape: input.shape)
  }

  static func metrics(
    reference: [Float],
    candidate: [Float]
  ) -> QuantizationErrorMetrics {
    var maximum = 0.0
    var sumSquares = 0.0
    for (expected, actual) in zip(reference, candidate) {
      let difference = Double(expected) - Double(actual)
      maximum = max(maximum, abs(difference))
      sumSquares += difference * difference
    }
    let rmse = reference.isEmpty ? 0 : sqrt(sumSquares / Double(reference.count))
    return QuantizationErrorMetrics(
      maximumAbsoluteError: Float(maximum), rootMeanSquareError: Float(rmse))
  }

  static func validateWeightMatrix(_ weights: FloatTensor, groupSize: Int) throws {
    guard weights.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: weights.rank)
    }
    guard groupSize > 0 else { throw WeightQuantizationError.invalidGroupSize(groupSize) }
    try validateFinite(weights.storage)
  }

  static func quantizeGroupwiseInt8(
    _ weights: FloatTensor,
    groupSize: Int
  ) throws -> GroupwiseInt8WeightMatrix {
    try validateWeightMatrix(weights, groupSize: groupSize)
    let outputChannels = weights.shape[0]
    let inputChannels = weights.shape[1]
    let groups = inputChannels == 0 ? 0 : (inputChannels + groupSize - 1) / groupSize
    var values = Array(repeating: Int8.zero, count: weights.elementCount)
    var scales = Array(repeating: Float.zero, count: outputChannels * groups)
    for output in 0..<outputChannels {
      for group in 0..<groups {
        let start = group * groupSize
        let end = min(start + groupSize, inputChannels)
        var maximum: Float = 0
        for input in start..<end {
          maximum = max(maximum, abs(weights.storage[output * inputChannels + input]))
        }
        let scale: Float = maximum == 0 ? 1 : maximum / 127
        scales[output * groups + group] = scale
        for input in start..<end {
          let source = weights.storage[output * inputChannels + input]
          let rounded = QuantizationRounding.nearestAwayFromZero(source / scale)
          values[output * inputChannels + input] = Int8(
            max(-127, min(127, Int(rounded))))
        }
      }
    }
    return try GroupwiseInt8WeightMatrix(
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      groupSize: groupSize,
      values: values,
      scales: scales)
  }

  static func dequantize(_ input: GroupwiseInt8WeightMatrix) throws -> FloatTensor {
    var values = Array(repeating: Float.zero, count: input.values.count)
    for output in 0..<input.outputChannels {
      for inputChannel in 0..<input.inputChannels {
        let index = output * input.inputChannels + inputChannel
        values[index] = Float(input.values[index])
          * input.scales[input.scaleIndex(
            outputChannel: output, inputChannel: inputChannel)]
      }
    }
    return try FloatTensor(values, shape: input.shape)
  }

  static func quantizeGroupwiseQ4(
    _ weights: FloatTensor,
    groupSize: Int
  ) throws -> GroupwiseQ4WeightMatrix {
    try validateWeightMatrix(weights, groupSize: groupSize)
    let outputChannels = weights.shape[0]
    let inputChannels = weights.shape[1]
    let groups = inputChannels == 0 ? 0 : (inputChannels + groupSize - 1) / groupSize
    var quantized = Array(repeating: Int8.zero, count: weights.elementCount)
    var scales = Array(repeating: Float.zero, count: outputChannels * groups)
    for output in 0..<outputChannels {
      for group in 0..<groups {
        let start = group * groupSize
        let end = min(start + groupSize, inputChannels)
        var maximum: Float = 0
        for input in start..<end {
          maximum = max(maximum, abs(weights.storage[output * inputChannels + input]))
        }
        let scale: Float = maximum == 0 ? 1 : maximum / 7
        scales[output * groups + group] = scale
        for input in start..<end {
          let source = weights.storage[output * inputChannels + input]
          let rounded = QuantizationRounding.nearestAwayFromZero(source / scale)
          quantized[output * inputChannels + input] = Int8(max(-8, min(7, Int(rounded))))
        }
      }
    }
    return try packQ4(
      quantized,
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      groupSize: groupSize,
      scales: scales)
  }

  static func dequantize(_ input: GroupwiseQ4WeightMatrix) throws -> FloatTensor {
    var values = Array(repeating: Float.zero, count: input.logicalValueCount)
    for output in 0..<input.outputChannels {
      for inputChannel in 0..<input.inputChannels {
        let index = output * input.inputChannels + inputChannel
        values[index] = Float(input.quantizedValue(
          outputChannel: output, inputChannel: inputChannel))
          * input.scales[input.scaleIndex(
            outputChannel: output, inputChannel: inputChannel)]
      }
    }
    return try FloatTensor(values, shape: input.shape)
  }

  static func packQ4(
    _ values: [Int8],
    outputChannels: Int,
    inputChannels: Int,
    groupSize: Int,
    scales: [Float]
  ) throws -> GroupwiseQ4WeightMatrix {
    let (logicalCount, overflow) = outputChannels.multipliedReportingOverflow(by: inputChannels)
    guard !overflow else { throw WeightQuantizationError.sizeOverflow }
    guard values.count == logicalCount else {
      throw WeightQuantizationError.storageCountMismatch(
        name: "Q4 logical values", expected: logicalCount, actual: values.count)
    }
    for (index, value) in values.enumerated()
      where value < GroupwiseQ4WeightMatrix.minimumStoredValue
        || value > GroupwiseQ4WeightMatrix.maximumStoredValue
    {
      throw WeightQuantizationError.quantizedValueOutOfRange(
        index: index,
        value: Int(value),
        minimum: GroupwiseQ4WeightMatrix.minimumStoredValue,
        maximum: GroupwiseQ4WeightMatrix.maximumStoredValue)
    }
    var packed = Array(repeating: UInt8.zero, count: logicalCount / 2 + logicalCount % 2)
    for (index, value) in values.enumerated() {
      let nibble = UInt8(bitPattern: value) & 0x0f
      if index.isMultiple(of: 2) {
        packed[index / 2] = nibble
      } else {
        packed[index / 2] |= nibble << 4
      }
    }
    return try GroupwiseQ4WeightMatrix(
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      groupSize: groupSize,
      packedValues: packed,
      scales: scales)
  }
}