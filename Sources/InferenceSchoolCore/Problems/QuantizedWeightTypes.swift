import Foundation

public enum WeightQuantizationError: Error, Equatable, LocalizedError {
  case invalidDimension(name: String, value: Int)
  case invalidGroupSize(Int)
  case storageCountMismatch(name: String, expected: Int, actual: Int)
  case nonFiniteValue(index: Int, value: Float)
  case invalidScale(index: Int, value: Float)
  case quantizedValueOutOfRange(index: Int, value: Int, minimum: Int, maximum: Int)
  case nonzeroPaddingNibble(UInt8)
  case shapeMismatch(name: String, expected: [Int], actual: [Int])
  case sizeOverflow

  public var errorDescription: String? {
    switch self {
    case .invalidDimension(let name, let value):
      "\(name) must be nonnegative; received \(value)."
    case .invalidGroupSize(let value):
      "Quantization group size must be positive; received \(value)."
    case .storageCountMismatch(let name, let expected, let actual):
      "\(name) requires \(expected) values; received \(actual)."
    case .nonFiniteValue(let index, let value):
      "Input value \(index) must be finite; received \(value)."
    case .invalidScale(let index, let value):
      "Scale \(index) must be finite and positive; received \(value)."
    case .quantizedValueOutOfRange(let index, let value, let minimum, let maximum):
      "Quantized value \(index) must be in \(minimum)...\(maximum); received \(value)."
    case .nonzeroPaddingNibble(let value):
      "The unused high nibble in an odd-length Q4 stream must be zero; received \(value)."
    case .shapeMismatch(let name, let expected, let actual):
      "\(name) must have shape \(expected); received \(actual)."
    case .sizeOverflow:
      "Quantized storage dimensions exceed Int.max."
    }
  }
}

public enum QuantizationRounding {
  public static func nearestAwayFromZero(_ value: Float) -> Float {
    value.rounded(.toNearestOrAwayFromZero)
  }
}

public struct QuantizationErrorMetrics: Sendable, Equatable {
  public let maximumAbsoluteError: Float
  public let rootMeanSquareError: Float

  public init(maximumAbsoluteError: Float, rootMeanSquareError: Float) {
    self.maximumAbsoluteError = maximumAbsoluteError
    self.rootMeanSquareError = rootMeanSquareError
  }
}

public struct SymmetricInt8Tensor: Sendable, Equatable {
  public static let minimumStoredValue: Int8 = -127
  public static let maximumStoredValue: Int8 = 127

  public let values: [Int8]
  public let shape: [Int]
  public let scale: Float

  public var elementCount: Int { values.count }
  public var allocatedBytes: Int {
    values.count * MemoryLayout<Int8>.stride + MemoryLayout<Float>.stride
  }

  public init(values: [Int8], shape: [Int], scale: Float) throws {
    let expected = try Self.elementCount(for: shape)
    guard values.count == expected else {
      throw WeightQuantizationError.storageCountMismatch(
        name: "INT8 tensor storage", expected: expected, actual: values.count)
    }
    guard scale.isFinite, scale > 0 else {
      throw WeightQuantizationError.invalidScale(index: 0, value: scale)
    }
    if let index = values.firstIndex(of: Int8.min) {
      throw WeightQuantizationError.quantizedValueOutOfRange(
        index: index, value: Int(Int8.min), minimum: -127, maximum: 127)
    }
    self.values = values
    self.shape = shape
    self.scale = scale
  }

  private static func elementCount(for shape: [Int]) throws -> Int {
    var count = 1
    for (axis, dimension) in shape.enumerated() {
      guard dimension >= 0 else {
        throw WeightQuantizationError.invalidDimension(name: "shape[\(axis)]", value: dimension)
      }
      let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
      guard !overflow else { throw WeightQuantizationError.sizeOverflow }
      count = next
    }
    return count
  }
}

public struct GroupwiseInt8WeightMatrix: Sendable, Equatable {
  public let outputChannels: Int
  public let inputChannels: Int
  public let groupSize: Int
  public let values: [Int8]
  public let scales: [Float]

  public var groupsPerOutputChannel: Int {
    inputChannels == 0 ? 0 : (inputChannels + groupSize - 1) / groupSize
  }
  public var shape: [Int] { [outputChannels, inputChannels] }
  public var valueBytes: Int { values.count * MemoryLayout<Int8>.stride }
  public var scaleBytes: Int { scales.count * MemoryLayout<Float>.stride }
  public var allocatedBytes: Int { valueBytes + scaleBytes }

  public init(
    outputChannels: Int,
    inputChannels: Int,
    groupSize: Int,
    values: [Int8],
    scales: [Float]
  ) throws {
    try validateWeightDimensions(
      outputChannels: outputChannels, inputChannels: inputChannels, groupSize: groupSize)
    let valueCount = try checkedProduct(outputChannels, inputChannels)
    let groups = inputChannels == 0 ? 0 : (inputChannels + groupSize - 1) / groupSize
    let scaleCount = try checkedProduct(outputChannels, groups)
    guard values.count == valueCount else {
      throw WeightQuantizationError.storageCountMismatch(
        name: "groupwise INT8 values", expected: valueCount, actual: values.count)
    }
    if let index = values.firstIndex(of: Int8.min) {
      throw WeightQuantizationError.quantizedValueOutOfRange(
        index: index, value: Int(Int8.min), minimum: -127, maximum: 127)
    }
    try validateScales(scales, expectedCount: scaleCount)
    self.outputChannels = outputChannels
    self.inputChannels = inputChannels
    self.groupSize = groupSize
    self.values = values
    self.scales = scales
  }

  public func scaleIndex(outputChannel: Int, inputChannel: Int) -> Int {
    outputChannel * groupsPerOutputChannel + inputChannel / groupSize
  }
}

public struct GroupwiseQ4WeightMatrix: Sendable, Equatable {
  public static let minimumStoredValue = -8
  public static let maximumStoredValue = 7

  public let format: Q4WeightFormat
  public let outputChannels: Int
  public let inputChannels: Int
  public let groupSize: Int
  public let packedValues: [UInt8]
  public let scales: [Float]

  public var groupsPerOutputChannel: Int {
    inputChannels == 0 ? 0 : (inputChannels + groupSize - 1) / groupSize
  }
  public var shape: [Int] { [outputChannels, inputChannels] }
  public var logicalValueCount: Int { outputChannels * inputChannels }
  public var packedValueBytes: Int { packedValues.count * MemoryLayout<UInt8>.stride }
  public var scaleBytes: Int { scales.count * MemoryLayout<Float>.stride }
  public var allocatedBytes: Int { packedValueBytes + scaleBytes }

  public init(
    format: Q4WeightFormat = .signedTwosComplementLowNibbleFirst,
    outputChannels: Int,
    inputChannels: Int,
    groupSize: Int,
    packedValues: [UInt8],
    scales: [Float]
  ) throws {
    try validateWeightDimensions(
      outputChannels: outputChannels, inputChannels: inputChannels, groupSize: groupSize)
    let valueCount = try checkedProduct(outputChannels, inputChannels)
    let packedCount = valueCount / 2 + valueCount % 2
    let groups = inputChannels == 0 ? 0 : (inputChannels + groupSize - 1) / groupSize
    let scaleCount = try checkedProduct(outputChannels, groups)
    guard packedValues.count == packedCount else {
      throw WeightQuantizationError.storageCountMismatch(
        name: "packed Q4 values", expected: packedCount, actual: packedValues.count)
    }
    if valueCount % 2 == 1, let last = packedValues.last, last & 0xf0 != 0 {
      throw WeightQuantizationError.nonzeroPaddingNibble(last >> 4)
    }
    try validateScales(scales, expectedCount: scaleCount)
    self.format = format
    self.outputChannels = outputChannels
    self.inputChannels = inputChannels
    self.groupSize = groupSize
    self.packedValues = packedValues
    self.scales = scales
  }

  public func scaleIndex(outputChannel: Int, inputChannel: Int) -> Int {
    outputChannel * groupsPerOutputChannel + inputChannel / groupSize
  }

  public func quantizedValue(outputChannel: Int, inputChannel: Int) -> Int8 {
    let logicalIndex = outputChannel * inputChannels + inputChannel
    let byte = packedValues[logicalIndex / 2]
    let nibble = logicalIndex.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
    return Int8(bitPattern: nibble >= 8 ? nibble | 0xf0 : nibble)
  }
}

public enum Q4WeightFormat: UInt32, Sendable, Equatable {
  case signedTwosComplementLowNibbleFirst = 0
}

private func validateWeightDimensions(
  outputChannels: Int,
  inputChannels: Int,
  groupSize: Int
) throws {
  guard outputChannels >= 0 else {
    throw WeightQuantizationError.invalidDimension(
      name: "outputChannels", value: outputChannels)
  }
  guard inputChannels >= 0 else {
    throw WeightQuantizationError.invalidDimension(name: "inputChannels", value: inputChannels)
  }
  guard groupSize > 0 else { throw WeightQuantizationError.invalidGroupSize(groupSize) }
}

private func checkedProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
  let (product, overflow) = lhs.multipliedReportingOverflow(by: rhs)
  guard !overflow else { throw WeightQuantizationError.sizeOverflow }
  return product
}

private func validateScales(_ scales: [Float], expectedCount: Int) throws {
  guard scales.count == expectedCount else {
    throw WeightQuantizationError.storageCountMismatch(
      name: "quantization scales", expected: expectedCount, actual: scales.count)
  }
  for (index, scale) in scales.enumerated() where !scale.isFinite || scale <= 0 {
    throw WeightQuantizationError.invalidScale(index: index, value: scale)
  }
}