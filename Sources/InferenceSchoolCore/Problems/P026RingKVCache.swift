import Foundation

public struct RingKVCacheRequest: Sendable, Equatable {
  public let configuration: KVCacheConfiguration
  public let layer: Int
  public let firstLogicalPosition: Int
  public let keys: FloatTensor
  public let values: FloatTensor
  public let query: FloatTensor
  public let queryHeadCount: Int
  public let window: Int

  public var tokenCount: Int { keys.shape.first ?? 0 }
  public var queryLogicalPosition: Int { firstLogicalPosition + tokenCount - 1 }

  public init(
    configuration: KVCacheConfiguration,
    layer: Int,
    firstLogicalPosition: Int,
    keys: FloatTensor,
    values: FloatTensor,
    query: FloatTensor,
    queryHeadCount: Int,
    window: Int
  ) throws {
    try configuration.validate(layer: layer)
    guard firstLogicalPosition >= 0 else {
      throw KVCacheError.invalidLogicalPosition(firstLogicalPosition)
    }
    let tokenCount = keys.shape.first ?? 0
    guard tokenCount > 0 else { throw KVCacheError.invalidTokenCount(tokenCount) }
    let expectedKV = [tokenCount, configuration.keyValueHeadCount, configuration.headDimension]
    guard keys.shape == expectedKV else {
      throw KVCacheError.vectorShapeMismatch(name: "Keys", expected: expectedKV, actual: keys.shape)
    }
    guard values.shape == expectedKV else {
      throw KVCacheError.vectorShapeMismatch(name: "Values", expected: expectedKV, actual: values.shape)
    }
    let attention = try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: configuration.keyValueHeadCount,
      headDimension: configuration.headDimension)
    let expectedQuery = [queryHeadCount, configuration.headDimension]
    guard query.shape == expectedQuery else {
      throw KVCacheError.vectorShapeMismatch(
        name: "Query", expected: expectedQuery, actual: query.shape)
    }
    guard window > 0 else { throw AttentionError.invalidWindow(window) }
    _ = attention
    self.configuration = configuration
    self.layer = layer
    self.firstLogicalPosition = firstLogicalPosition
    self.keys = keys
    self.values = values
    self.query = query
    self.queryHeadCount = queryHeadCount
    self.window = window
  }
}

public struct RingKVCacheResult: Sendable, Equatable {
  public let chronologicalHistory: [[Int]]
  public let finalSnapshot: KVCacheLayerSnapshot
  public let attentionOutput: FloatTensor
  public let allocatedBytes: Int
  public let storageAddressesStable: Bool

  public init(
    chronologicalHistory: [[Int]],
    finalSnapshot: KVCacheLayerSnapshot,
    attentionOutput: FloatTensor,
    allocatedBytes: Int,
    storageAddressesStable: Bool
  ) {
    self.chronologicalHistory = chronologicalHistory
    self.finalSnapshot = finalSnapshot
    self.attentionOutput = attentionOutput
    self.allocatedBytes = allocatedBytes
    self.storageAddressesStable = storageAddressesStable
  }
}

public typealias RingKVCacheImplementation = (
  _ request: RingKVCacheRequest
) throws -> RingKVCacheResult

public enum P026RingKVCacheJudge {
  public static func evaluate(_ implementation: RingKVCacheImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let configuration = try KVCacheConfiguration(
        layerCount: 2, keyValueHeadCount: 1, headDimension: 2, capacity: 3)
      let keys = try FloatTensor(
        (0..<16).map { Float($0 + 1) / 8 }, shape: [8, 1, 2])
      let values = try FloatTensor(
        (0..<16).map { Float($0 + 20) / 7 }, shape: [8, 1, 2])
      let request = try RingKVCacheRequest(
        configuration: configuration,
        layer: 1,
        firstLogicalPosition: 10,
        keys: keys,
        values: values,
        query: FloatTensor([0.75, -0.25], shape: [1, 2]),
        queryHeadCount: 1,
        window: 2)
      let actual = try implementation(request)
      let expectedHistory = [
        [10], [10, 11], [10, 11, 12], [11, 12, 13],
        [12, 13, 14], [13, 14, 15], [14, 15, 16], [15, 16, 17],
      ]
      if actual.chronologicalHistory == expectedHistory,
        actual.finalSnapshot.logicalPositions == [15, 16, 17]
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "chronological reads across repeated wraps",
          message: "logical positions did not remain monotonic after physical overwrite"))
      }
      let expectedKeys = try FloatTensor(Array(keys.storage.suffix(6)), shape: [3, 1, 2])
      let expectedValues = try FloatTensor(Array(values.storage.suffix(6)), shape: [3, 1, 2])
      if actual.finalSnapshot.keys == expectedKeys, actual.finalSnapshot.values == expectedValues,
        actual.allocatedBytes == configuration.allocatedFloat32Bytes,
        actual.storageAddressesStable
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "overwrite keeps fixed storage and newest values",
          message: "ring contents or fixed-allocation evidence is incorrect"))
      }
      let oracleConfiguration = try AttentionConfiguration(
        queryHeadCount: 1,
        keyValueHeadCount: 1,
        headDimension: 2,
        queryPositionOffset: 17,
        keyPositionOffset: 15)
      let oracleQuery = try FloatTensor(request.query.storage, shape: [1, 1, 2])
      let expectedOutput3D = try AttentionJudgeOracle.materialized(
        queries: oracleQuery,
        keys: expectedKeys,
        values: expectedValues,
        configuration: oracleConfiguration,
        window: 2)
      let expectedOutput = try FloatTensor(expectedOutput3D.storage, shape: [1, 2])
      if AttentionJudgeOracle.approximatelyEqual(actual.attentionOutput, expectedOutput) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "windowed attention reads chronological ring view",
          message: "attention used physical slot order or stale overwritten values"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "ring cache", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 3, failures: failures)
  }
}
