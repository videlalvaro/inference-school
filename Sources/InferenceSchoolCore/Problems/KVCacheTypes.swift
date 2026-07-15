import Foundation

public enum KVCacheError: Error, Equatable, LocalizedError {
  case invalidLayerCount(Int)
  case invalidKeyValueHeadCount(Int)
  case invalidHeadDimension(Int)
  case invalidCapacity(Int)
  case storageSizeOverflow
  case layerOutOfBounds(layer: Int, layerCount: Int)
  case headOutOfBounds(head: Int, headCount: Int)
  case invalidLogicalPosition(Int)
  case invalidTokenCount(Int)
  case slotOutOfBounds(slot: Int, capacity: Int)
  case positionSequenceMismatch(layer: Int, expected: Int, actual: Int)
  case capacityExceeded(layer: Int, capacity: Int)
  case vectorShapeMismatch(name: String, expected: [Int], actual: [Int])
  case positionNotCached(layer: Int, logicalPosition: Int)
  case pageUnavailable
  case invalidPageSize(Int)
  case invalidPageCount(Int)
  case invalidQuantizationBlockSize(Int)

  public var errorDescription: String? {
    switch self {
    case .invalidLayerCount(let value):
      "Layer count must be positive; received \(value)."
    case .invalidKeyValueHeadCount(let value):
      "Key/value head count must be positive; received \(value)."
    case .invalidHeadDimension(let value):
      "Head dimension must be positive; received \(value)."
    case .invalidCapacity(let value):
      "Cache capacity must be positive; received \(value)."
    case .storageSizeOverflow:
      "The cache storage size exceeds Int.max."
    case .layerOutOfBounds(let layer, let layerCount):
      "Layer \(layer) is outside 0..<\(layerCount)."
    case .headOutOfBounds(let head, let headCount):
      "KV head \(head) is outside 0..<\(headCount)."
    case .invalidLogicalPosition(let position):
      "Logical token position must be nonnegative; received \(position)."
    case .invalidTokenCount(let value):
      "Token count must be positive; received \(value)."
    case .slotOutOfBounds(let slot, let capacity):
      "Cache slot \(slot) is outside 0..<\(capacity)."
    case .positionSequenceMismatch(let layer, let expected, let actual):
      "Layer \(layer) expected logical position \(expected); received \(actual)."
    case .capacityExceeded(let layer, let capacity):
      "Layer \(layer) has reached its fixed capacity of \(capacity) tokens."
    case .vectorShapeMismatch(let name, let expected, let actual):
      "\(name) must have shape \(expected); received \(actual)."
    case .positionNotCached(let layer, let logicalPosition):
      "Layer \(layer) does not contain logical position \(logicalPosition)."
    case .pageUnavailable:
      "No free KV-cache page is available."
    case .invalidPageSize(let value):
      "Page size must be positive; received \(value)."
    case .invalidPageCount(let value):
      "Physical page count must be positive; received \(value)."
    case .invalidQuantizationBlockSize(let value):
      "Quantization block size must be positive; received \(value)."
    }
  }
}

public struct KVCacheConfiguration: Sendable, Equatable {
  public let layerCount: Int
  public let keyValueHeadCount: Int
  public let headDimension: Int
  public let capacity: Int

  public var batchSize: Int { 1 }
  public var elementsPerToken: Int { keyValueHeadCount * headDimension }
  public var elementsPerTensor: Int { layerCount * capacity * elementsPerToken }
  public var allocatedFloat32Bytes: Int {
    2 * elementsPerTensor * MemoryLayout<Float>.stride
  }

  public init(
    layerCount: Int,
    keyValueHeadCount: Int,
    headDimension: Int,
    capacity: Int
  ) throws {
    guard layerCount > 0 else { throw KVCacheError.invalidLayerCount(layerCount) }
    guard keyValueHeadCount > 0 else {
      throw KVCacheError.invalidKeyValueHeadCount(keyValueHeadCount)
    }
    guard headDimension > 0 else { throw KVCacheError.invalidHeadDimension(headDimension) }
    guard capacity > 0 else { throw KVCacheError.invalidCapacity(capacity) }

    var elementCount = 1
    for dimension in [layerCount, capacity, keyValueHeadCount, headDimension] {
      let (next, overflow) = elementCount.multipliedReportingOverflow(by: dimension)
      guard !overflow else { throw KVCacheError.storageSizeOverflow }
      elementCount = next
    }
    let (bothTensors, doubleOverflow) = elementCount.multipliedReportingOverflow(by: 2)
    let (_, byteOverflow) = bothTensors.multipliedReportingOverflow(
      by: MemoryLayout<Float>.stride)
    guard !doubleOverflow, !byteOverflow else { throw KVCacheError.storageSizeOverflow }

    self.layerCount = layerCount
    self.keyValueHeadCount = keyValueHeadCount
    self.headDimension = headDimension
    self.capacity = capacity
  }

  public func validate(layer: Int) throws {
    guard layer >= 0, layer < layerCount else {
      throw KVCacheError.layerOutOfBounds(layer: layer, layerCount: layerCount)
    }
  }

  public func validate(head: Int) throws {
    guard head >= 0, head < keyValueHeadCount else {
      throw KVCacheError.headOutOfBounds(head: head, headCount: keyValueHeadCount)
    }
  }

  public func validate(vector: FloatTensor, name: String) throws {
    let expected = [keyValueHeadCount, headDimension]
    guard vector.shape == expected else {
      throw KVCacheError.vectorShapeMismatch(name: name, expected: expected, actual: vector.shape)
    }
  }
}

public protocol KVCacheReadable: AnyObject {
  var configuration: KVCacheConfiguration { get }
  var allocatedBytes: Int { get }

  func count(layer: Int) throws -> Int
  func logicalPositions(layer: Int) throws -> [Int]
  func keyVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float]
  func valueVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float]
}

public protocol KVCacheWritable: KVCacheReadable {
  func append(layer: Int, logicalPosition: Int, key: FloatTensor, value: FloatTensor) throws
}

public final class ContiguousKVCache: KVCacheWritable {
  public let configuration: KVCacheConfiguration
  public var allocatedBytes: Int { configuration.allocatedFloat32Bytes }
  public var keyStorageCount: Int { keyStorage.count }
  public var valueStorageCount: Int { valueStorage.count }

  private var keyStorage: [Float]
  private var valueStorage: [Float]
  private var positions: [Int]
  private var counts: [Int]
  private var firstPositions: [Int?]

  public init(configuration: KVCacheConfiguration) {
    self.configuration = configuration
    keyStorage = Array(repeating: 0, count: configuration.elementsPerTensor)
    valueStorage = Array(repeating: 0, count: configuration.elementsPerTensor)
    positions = Array(repeating: -1, count: configuration.layerCount * configuration.capacity)
    counts = Array(repeating: 0, count: configuration.layerCount)
    firstPositions = Array(repeating: nil, count: configuration.layerCount)
  }

  public func storageAddresses() -> (key: UInt, value: UInt) {
    let key = keyStorage.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) }
    let value = valueStorage.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) }
    return (key, value)
  }

  public func count(layer: Int) throws -> Int {
    try configuration.validate(layer: layer)
    return counts[layer]
  }

  public func logicalPositions(layer: Int) throws -> [Int] {
    try configuration.validate(layer: layer)
    let start = layer * configuration.capacity
    return Array(positions[start..<(start + counts[layer])])
  }

  public func append(
    layer: Int,
    logicalPosition: Int,
    key: FloatTensor,
    value: FloatTensor
  ) throws {
    let record = KVCacheAppend(
      layer: layer, logicalPosition: logicalPosition, key: key, value: value)
    let lastPositions = firstPositions.enumerated().map { index, first in
      first.map { $0 + counts[index] - 1 }
    }
    try P022ContiguousKVCacheContract.validate(
      record, configuration: configuration, counts: counts, lastPositions: lastPositions)

    let slot = counts[layer]
    let destination = offset(layer: layer, slot: slot, head: 0, feature: 0)
    for index in 0..<configuration.elementsPerToken {
      keyStorage[destination + index] = key.storage[index]
      valueStorage[destination + index] = value.storage[index]
    }
    positions[layer * configuration.capacity + slot] = logicalPosition
    if firstPositions[layer] == nil { firstPositions[layer] = logicalPosition }
    counts[layer] += 1
  }

  public func keyVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try vector(
      storage: keyStorage, layer: layer, logicalPosition: logicalPosition, head: head)
  }

  public func valueVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try vector(
      storage: valueStorage, layer: layer, logicalPosition: logicalPosition, head: head)
  }

  public func rawKeyStorage() -> [Float] { keyStorage }
  public func rawValueStorage() -> [Float] { valueStorage }

  private func vector(
    storage: [Float], layer: Int, logicalPosition: Int, head: Int
  ) throws -> [Float] {
    try configuration.validate(layer: layer)
    try configuration.validate(head: head)
    guard let first = firstPositions[layer] else {
      throw KVCacheError.positionNotCached(layer: layer, logicalPosition: logicalPosition)
    }
    let slot = logicalPosition - first
    guard slot >= 0, slot < counts[layer],
      positions[layer * configuration.capacity + slot] == logicalPosition
    else {
      throw KVCacheError.positionNotCached(layer: layer, logicalPosition: logicalPosition)
    }
    let start = offset(layer: layer, slot: slot, head: head, feature: 0)
    return Array(storage[start..<(start + configuration.headDimension)])
  }

  private func offset(layer: Int, slot: Int, head: Int, feature: Int) -> Int {
    (((layer * configuration.capacity) + slot) * configuration.keyValueHeadCount + head)
      * configuration.headDimension + feature
  }
}

public extension KVCacheReadable {
  func materialized(layer: Int) throws -> (
    positions: [Int], keys: FloatTensor, values: FloatTensor
  ) {
    let positions = try logicalPositions(layer: layer)
    var keys: [Float] = []
    var values: [Float] = []
    keys.reserveCapacity(positions.count * configuration.elementsPerToken)
    values.reserveCapacity(positions.count * configuration.elementsPerToken)
    for position in positions {
      for head in 0..<configuration.keyValueHeadCount {
        keys.append(contentsOf: try keyVector(
          layer: layer, logicalPosition: position, head: head))
        values.append(contentsOf: try valueVector(
          layer: layer, logicalPosition: position, head: head))
      }
    }
    return (
      positions,
      try FloatTensor(
        keys,
        shape: [positions.count, configuration.keyValueHeadCount, configuration.headDimension]),
      try FloatTensor(
        values,
        shape: [positions.count, configuration.keyValueHeadCount, configuration.headDimension])
    )
  }
}

public struct KVCacheAppend: Sendable, Equatable {
  public let layer: Int
  public let logicalPosition: Int
  public let key: FloatTensor
  public let value: FloatTensor

  public init(layer: Int, logicalPosition: Int, key: FloatTensor, value: FloatTensor) {
    self.layer = layer
    self.logicalPosition = logicalPosition
    self.key = key
    self.value = value
  }
}

public struct KVCacheLayerSnapshot: Sendable, Equatable {
  public let logicalPositions: [Int]
  public let keys: FloatTensor
  public let values: FloatTensor

  public init(logicalPositions: [Int], keys: FloatTensor, values: FloatTensor) {
    self.logicalPositions = logicalPositions
    self.keys = keys
    self.values = values
  }
}

public struct ContiguousKVCacheTranscript: Sendable, Equatable {
  public let layers: [KVCacheLayerSnapshot]
  public let allocatedBytes: Int
  public let keyStorageCount: Int
  public let valueStorageCount: Int
  public let storageAddressesStable: Bool

  public init(
    layers: [KVCacheLayerSnapshot],
    allocatedBytes: Int,
    keyStorageCount: Int,
    valueStorageCount: Int,
    storageAddressesStable: Bool
  ) {
    self.layers = layers
    self.allocatedBytes = allocatedBytes
    self.keyStorageCount = keyStorageCount
    self.valueStorageCount = valueStorageCount
    self.storageAddressesStable = storageAddressesStable
  }
}

public typealias ContiguousKVCacheImplementation = (
  _ configuration: KVCacheConfiguration,
  _ appends: [KVCacheAppend]
) throws -> ContiguousKVCacheTranscript
