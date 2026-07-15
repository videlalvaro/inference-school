import Foundation
import InferenceSchoolCore

public enum P036WeightContainerSolution {
  public static func parse(
    bytes: [UInt8],
    requiredTensorNames: [String]
  ) throws -> ParsedWeightContainer {
    let preamble = try P036WeightContainerContract.validatePreamble(bytes)
    let headerBytes = Array(
      bytes[P036WeightContainerContract.preambleByteCount..<preamble.headerEnd])
    guard String(bytes: headerBytes, encoding: .utf8) != nil else {
      throw WeightContainerError.invalidUTF8Header
    }
    let header: WeightContainerHeader
    do {
      header = try JSONDecoder().decode(WeightContainerHeader.self, from: Data(headerBytes))
    } catch {
      throw WeightContainerError.invalidJSONHeader(error.localizedDescription)
    }
    let configuration = try header.model.decoderConfiguration()
    let payloadSize = bytes.count - preamble.payloadStart
    var names: Set<String> = []
    var metadata: [String: WeightTensorMetadata] = [:]
    var ranges: [PayloadRange] = []

    for (index, descriptor) in header.tensors.enumerated() {
      guard !descriptor.name.isEmpty else {
        throw WeightContainerError.emptyTensorName(index: index)
      }
      guard names.insert(descriptor.name).inserted else {
        throw WeightContainerError.duplicateTensorName(descriptor.name)
      }
      guard descriptor.dtype == P036WeightContainerContract.float32DType else {
        throw WeightContainerError.unsupportedDType(
          tensor: descriptor.name, dtype: descriptor.dtype)
      }
      guard descriptor.offset >= 0 else {
        throw WeightContainerError.negativeOffset(
          tensor: descriptor.name, value: descriptor.offset)
      }
      guard descriptor.offset.isMultiple(
        of: P036WeightContainerContract.float32Alignment)
      else {
        throw WeightContainerError.misalignedOffset(
          tensor: descriptor.name,
          offset: descriptor.offset,
          alignment: P036WeightContainerContract.float32Alignment)
      }
      guard descriptor.byteCount >= 0 else {
        throw WeightContainerError.negativeByteCount(
          tensor: descriptor.name, value: descriptor.byteCount)
      }
      let elementCount = try checkedElementCount(
        shape: descriptor.shape, tensor: descriptor.name)
      let expectedBytes = try checkedMultiply(
        elementCount, MemoryLayout<Float>.size, tensor: descriptor.name)
      guard descriptor.byteCount == expectedBytes else {
        throw WeightContainerError.byteCountMismatch(
          tensor: descriptor.name,
          expected: expectedBytes,
          actual: descriptor.byteCount)
      }
      let (end, overflow) = descriptor.offset.addingReportingOverflow(descriptor.byteCount)
      guard !overflow, end <= payloadSize else {
        throw WeightContainerError.payloadOutOfBounds(
          tensor: descriptor.name,
          end: overflow ? Int.max : end,
          payloadSize: payloadSize)
      }
      ranges.append(PayloadRange(name: descriptor.name, start: descriptor.offset, end: end))
      metadata[descriptor.name] = descriptor
    }

    let nonemptyRanges = ranges.filter { $0.start < $0.end }.sorted {
      $0.start == $1.start ? $0.name < $1.name : $0.start < $1.start
    }
    if nonemptyRanges.count > 1 {
      for index in 1..<nonemptyRanges.count {
        if nonemptyRanges[index].start < nonemptyRanges[index - 1].end {
          throw WeightContainerError.overlappingTensors(
            first: nonemptyRanges[index - 1].name,
            second: nonemptyRanges[index].name)
        }
      }
    }

    var tensors: [String: FloatTensor] = [:]
    for descriptor in header.tensors {
      let absoluteStart = preamble.payloadStart + descriptor.offset
      var values: [Float] = []
      values.reserveCapacity(descriptor.byteCount / MemoryLayout<Float>.size)
      for byteOffset in stride(from: 0, to: descriptor.byteCount, by: 4) {
        let bits = readUInt32LittleEndian(bytes, at: absoluteStart + byteOffset)
        let value = Float(bitPattern: bits)
        guard value.isFinite else {
          throw WeightContainerError.nonFiniteTensorValue(
            tensor: descriptor.name, linearIndex: byteOffset / 4)
        }
        values.append(value)
      }
      tensors[descriptor.name] = try FloatTensor(values, shape: descriptor.shape)
    }
    for name in requiredTensorNames where tensors[name] == nil {
      throw WeightContainerError.missingTensor(name)
    }
    return ParsedWeightContainer(
      configuration: configuration,
      tensorMetadata: metadata,
      tensors: tensors,
      payloadByteCount: payloadSize)
  }

  private struct PayloadRange {
    let name: String
    let start: Int
    let end: Int
  }

  private static func checkedElementCount(shape: [Int], tensor: String) throws -> Int {
    var count = 1
    for (axis, dimension) in shape.enumerated() {
      guard dimension >= 0 else {
        throw WeightContainerError.invalidDimension(
          tensor: tensor, axis: axis, value: dimension)
      }
      count = try checkedMultiply(count, dimension, tensor: tensor)
    }
    return count
  }

  private static func checkedMultiply(_ lhs: Int, _ rhs: Int, tensor: String) throws -> Int {
    let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw WeightContainerError.tensorSizeOverflow(tensor) }
    return result
  }

  private static func readUInt32LittleEndian(_ bytes: [UInt8], at offset: Int) -> UInt32 {
    UInt32(bytes[offset])
      | UInt32(bytes[offset + 1]) << 8
      | UInt32(bytes[offset + 2]) << 16
      | UInt32(bytes[offset + 3]) << 24
  }
}