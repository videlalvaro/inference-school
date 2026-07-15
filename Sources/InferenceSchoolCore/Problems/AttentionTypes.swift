import Foundation

public enum AttentionError: Error, Equatable, LocalizedError {
  case invalidHeadCount(name: String, value: Int)
  case invalidHeadDimension(Int)
  case queryHeadsNotDivisible(queryHeads: Int, keyValueHeads: Int)
  case invalidPositionOffset(name: String, value: Int)
  case shapeMismatch(tensor: String, expected: [Int], actual: [Int])
  case keyValueLengthMismatch(keys: Int, values: Int)
  case nonFiniteValue(tensor: String, linearIndex: Int)
  case invalidWindow(Int)
  case noVisibleKeys(queryPosition: Int)
  case unsupportedSequenceLength(maximum: Int, actual: Int)
  case unsupportedHeadDimension(maximum: Int, actual: Int)

  public var errorDescription: String? {
    switch self {
    case .invalidHeadCount(let name, let value):
      "\(name) must be positive; received \(value)."
    case .invalidHeadDimension(let value):
      "Head dimension must be positive; received \(value)."
    case .queryHeadsNotDivisible(let queryHeads, let keyValueHeads):
      "Query heads (\(queryHeads)) must be divisible by key/value heads (\(keyValueHeads))."
    case .invalidPositionOffset(let name, let value):
      "\(name) must be nonnegative; received \(value)."
    case .shapeMismatch(let tensor, let expected, let actual):
      "\(tensor) must have shape \(expected); received \(actual)."
    case .keyValueLengthMismatch(let keys, let values):
      "Key and value sequence lengths must match; received \(keys) and \(values)."
    case .nonFiniteValue(let tensor, let linearIndex):
      "\(tensor) contains a non-finite value at linear index \(linearIndex)."
    case .invalidWindow(let value):
      "Attention window must be positive; received \(value)."
    case .noVisibleKeys(let queryPosition):
      "Query position \(queryPosition) has no visible key positions."
    case .unsupportedSequenceLength(let maximum, let actual):
      "This Metal kernel supports sequence lengths up to \(maximum); received \(actual)."
    case .unsupportedHeadDimension(let maximum, let actual):
      "This Metal kernel supports head dimensions up to \(maximum); received \(actual)."
    }
  }
}

public struct AttentionConfiguration: Sendable, Equatable {
  public let queryHeadCount: Int
  public let keyValueHeadCount: Int
  public let headDimension: Int
  public let queryPositionOffset: Int
  public let keyPositionOffset: Int

  public var groupSize: Int { queryHeadCount / keyValueHeadCount }

  public init(
    queryHeadCount: Int,
    keyValueHeadCount: Int,
    headDimension: Int,
    queryPositionOffset: Int = 0,
    keyPositionOffset: Int = 0
  ) throws {
    guard queryHeadCount > 0 else {
      throw AttentionError.invalidHeadCount(name: "Query head count", value: queryHeadCount)
    }
    guard keyValueHeadCount > 0 else {
      throw AttentionError.invalidHeadCount(name: "Key/value head count", value: keyValueHeadCount)
    }
    guard headDimension > 0 else {
      throw AttentionError.invalidHeadDimension(headDimension)
    }
    guard queryHeadCount.isMultiple(of: keyValueHeadCount) else {
      throw AttentionError.queryHeadsNotDivisible(
        queryHeads: queryHeadCount,
        keyValueHeads: keyValueHeadCount
      )
    }
    guard queryPositionOffset >= 0 else {
      throw AttentionError.invalidPositionOffset(
        name: "Query position offset", value: queryPositionOffset)
    }
    guard keyPositionOffset >= 0 else {
      throw AttentionError.invalidPositionOffset(
        name: "Key position offset", value: keyPositionOffset)
    }

    self.queryHeadCount = queryHeadCount
    self.keyValueHeadCount = keyValueHeadCount
    self.headDimension = headDimension
    self.queryPositionOffset = queryPositionOffset
    self.keyPositionOffset = keyPositionOffset
  }

  public func keyValueHead(forQueryHead queryHead: Int) -> Int {
    queryHead / groupSize
  }
}

public struct AttentionInput: Sendable, Equatable {
  public let queries: FloatTensor
  public let keys: FloatTensor
  public let values: FloatTensor
  public let configuration: AttentionConfiguration

  public var queryLength: Int { queries.shape[0] }
  public var keyValueLength: Int { keys.shape[0] }

  public init(
    queries: FloatTensor,
    keys: FloatTensor,
    values: FloatTensor,
    configuration: AttentionConfiguration
  ) throws {
    let expectedQueryShape = [
      queries.shape.first ?? 0,
      configuration.queryHeadCount,
      configuration.headDimension,
    ]
    let expectedKeyShape = [
      keys.shape.first ?? 0,
      configuration.keyValueHeadCount,
      configuration.headDimension,
    ]
    let expectedValueShape = [
      values.shape.first ?? 0,
      configuration.keyValueHeadCount,
      configuration.headDimension,
    ]

    guard queries.rank == 3, queries.shape == expectedQueryShape else {
      throw AttentionError.shapeMismatch(
        tensor: "Queries",
        expected: expectedQueryShape,
        actual: queries.shape
      )
    }
    guard keys.rank == 3, keys.shape == expectedKeyShape else {
      throw AttentionError.shapeMismatch(
        tensor: "Keys",
        expected: expectedKeyShape,
        actual: keys.shape
      )
    }
    guard values.rank == 3, values.shape == expectedValueShape else {
      throw AttentionError.shapeMismatch(
        tensor: "Values",
        expected: expectedValueShape,
        actual: values.shape
      )
    }
    guard keys.shape[0] == values.shape[0] else {
      throw AttentionError.keyValueLengthMismatch(keys: keys.shape[0], values: values.shape[0])
    }

    for (tensorName, tensor) in [("Queries", queries), ("Keys", keys), ("Values", values)] {
      if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
        throw AttentionError.nonFiniteValue(tensor: tensorName, linearIndex: index)
      }
    }

    self.queries = queries
    self.keys = keys
    self.values = values
    self.configuration = configuration
  }

  public func queryOffset(sequence: Int, head: Int, feature: Int) -> Int {
    (sequence * configuration.queryHeadCount + head) * configuration.headDimension + feature
  }

  public func keyValueOffset(sequence: Int, head: Int, feature: Int) -> Int {
    (sequence * configuration.keyValueHeadCount + head) * configuration.headDimension + feature
  }
}

public typealias AttentionImplementation = (
  _ queries: FloatTensor,
  _ keys: FloatTensor,
  _ values: FloatTensor,
  _ configuration: AttentionConfiguration
) throws -> FloatTensor
public typealias WindowedAttentionImplementation = (
  _ queries: FloatTensor,
  _ keys: FloatTensor,
  _ values: FloatTensor,
  _ configuration: AttentionConfiguration,
  _ window: Int
) throws -> FloatTensor
