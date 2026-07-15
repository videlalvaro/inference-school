import Foundation

public struct CachedAttentionRequest: Sendable, Equatable {
  public let cacheConfiguration: KVCacheConfiguration
  public let attentionConfiguration: AttentionConfiguration
  public let layer: Int
  public let firstLogicalPosition: Int
  public let queryLogicalPosition: Int
  public let query: FloatTensor
  public let keys: FloatTensor
  public let values: FloatTensor

  public var tokenCount: Int { keys.shape.first ?? 0 }

  public init(
    cacheConfiguration: KVCacheConfiguration,
    queryHeadCount: Int,
    layer: Int,
    firstLogicalPosition: Int,
    queryLogicalPosition: Int,
    query: FloatTensor,
    keys: FloatTensor,
    values: FloatTensor
  ) throws {
    try cacheConfiguration.validate(layer: layer)
    guard firstLogicalPosition >= 0 else {
      throw KVCacheError.invalidLogicalPosition(firstLogicalPosition)
    }
    guard queryLogicalPosition >= 0 else {
      throw KVCacheError.invalidLogicalPosition(queryLogicalPosition)
    }
    let tokenCount = keys.shape.first ?? 0
    guard tokenCount > 0 else { throw KVCacheError.invalidTokenCount(tokenCount) }
    guard tokenCount <= cacheConfiguration.capacity else {
      throw KVCacheError.capacityExceeded(layer: layer, capacity: cacheConfiguration.capacity)
    }
    let attentionConfiguration = try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: cacheConfiguration.keyValueHeadCount,
      headDimension: cacheConfiguration.headDimension,
      queryPositionOffset: queryLogicalPosition,
      keyPositionOffset: firstLogicalPosition)
    let expectedQuery = [queryHeadCount, cacheConfiguration.headDimension]
    guard query.shape == expectedQuery else {
      throw KVCacheError.vectorShapeMismatch(
        name: "Query", expected: expectedQuery, actual: query.shape)
    }
    let expectedCache = [
      tokenCount, cacheConfiguration.keyValueHeadCount, cacheConfiguration.headDimension,
    ]
    guard keys.shape == expectedCache else {
      throw KVCacheError.vectorShapeMismatch(
        name: "Keys", expected: expectedCache, actual: keys.shape)
    }
    guard values.shape == expectedCache else {
      throw KVCacheError.vectorShapeMismatch(
        name: "Values", expected: expectedCache, actual: values.shape)
    }
    let expectedPosition = firstLogicalPosition + tokenCount - 1
    guard queryLogicalPosition == expectedPosition else {
      throw KVCacheError.positionSequenceMismatch(
        layer: layer, expected: expectedPosition, actual: queryLogicalPosition)
    }
    for (name, tensor) in [("Query", query), ("Keys", keys), ("Values", values)] {
      if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
        throw AttentionError.nonFiniteValue(tensor: name, linearIndex: index)
      }
    }

    self.cacheConfiguration = cacheConfiguration
    self.attentionConfiguration = attentionConfiguration
    self.layer = layer
    self.firstLogicalPosition = firstLogicalPosition
    self.queryLogicalPosition = queryLogicalPosition
    self.query = query
    self.keys = keys
    self.values = values
  }
}

public struct CachedAttentionResult: Sendable, Equatable {
  public let output: FloatTensor
  public let cachedLogicalPositions: [Int]
  public let allocatedBytes: Int

  public init(output: FloatTensor, cachedLogicalPositions: [Int], allocatedBytes: Int) {
    self.output = output
    self.cachedLogicalPositions = cachedLogicalPositions
    self.allocatedBytes = allocatedBytes
  }
}

public typealias CachedAttentionImplementation = (
  _ request: CachedAttentionRequest
) throws -> CachedAttentionResult

public enum P023CachedAttentionJudge {
  public static func evaluate(_ implementation: CachedAttentionImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      for request in try fixtures() {
        let actual = try implementation(request)
        let expected = try oracle(request)
        if AttentionJudgeOracle.approximatelyEqual(actual.output, expected),
          actual.cachedLogicalPositions
            == Array(request.firstLogicalPosition...request.queryLogicalPosition),
          actual.allocatedBytes == request.cacheConfiguration.allocatedFloat32Bytes
        {
          passed += 1
        } else {
          failures.append(JudgeFailure(
            caseName: request.firstLogicalPosition == 0
              ? "append current token then decode" : "absolute positions remain logical",
            message: "cached decode differs from the materialized Double oracle or cache transcript"))
        }
      }
    } catch {
      failures.append(JudgeFailure(caseName: "valid cached decode", message: error.localizedDescription))
    }

    passed += expectError(name: "reject non-current query position", failures: &failures) {
      let cache = try KVCacheConfiguration(
        layerCount: 1, keyValueHeadCount: 1, headDimension: 2, capacity: 3)
      _ = try CachedAttentionRequest(
        cacheConfiguration: cache,
        queryHeadCount: 1,
        layer: 0,
        firstLogicalPosition: 4,
        queryLogicalPosition: 7,
        query: FloatTensor([1, 0], shape: [1, 2]),
        keys: FloatTensor([1, 0, 0, 1], shape: [2, 1, 2]),
        values: FloatTensor([1, 2, 3, 4], shape: [2, 1, 2]))
    }
    passed += expectError(name: "reject query shape", failures: &failures) {
      let cache = try KVCacheConfiguration(
        layerCount: 1, keyValueHeadCount: 1, headDimension: 2, capacity: 1)
      _ = try CachedAttentionRequest(
        cacheConfiguration: cache,
        queryHeadCount: 1,
        layer: 0,
        firstLogicalPosition: 0,
        queryLogicalPosition: 0,
        query: FloatTensor([1], shape: [1, 1]),
        keys: FloatTensor([1, 0], shape: [1, 1, 2]),
        values: FloatTensor([1, 2], shape: [1, 1, 2]))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }

  private static func fixtures() throws -> [CachedAttentionRequest] {
    let oneTokenCache = try KVCacheConfiguration(
      layerCount: 1, keyValueHeadCount: 2, headDimension: 2, capacity: 4)
    let offsetCache = try KVCacheConfiguration(
      layerCount: 2, keyValueHeadCount: 2, headDimension: 3, capacity: 4)
    return [
      try CachedAttentionRequest(
        cacheConfiguration: oneTokenCache,
        queryHeadCount: 2,
        layer: 0,
        firstLogicalPosition: 0,
        queryLogicalPosition: 0,
        query: FloatTensor([1, 2, 3, 4], shape: [2, 2]),
        keys: FloatTensor([2, 1, 4, 3], shape: [1, 2, 2]),
        values: FloatTensor([5, 6, 7, 8], shape: [1, 2, 2])),
      try CachedAttentionRequest(
        cacheConfiguration: offsetCache,
        queryHeadCount: 2,
        layer: 1,
        firstLogicalPosition: 11,
        queryLogicalPosition: 13,
        query: FloatTensor(attentionValues(count: 6, salt: 9), shape: [2, 3]),
        keys: FloatTensor(attentionValues(count: 18, salt: 5), shape: [3, 2, 3]),
        values: FloatTensor(attentionValues(count: 18, salt: 7), shape: [3, 2, 3])),
    ]
  }

  static func oracle(_ request: CachedAttentionRequest) throws -> FloatTensor {
    let query = try FloatTensor(
      request.query.storage,
      shape: [1, request.attentionConfiguration.queryHeadCount,
        request.cacheConfiguration.headDimension])
    let output = try AttentionJudgeOracle.materialized(
      queries: query,
      keys: request.keys,
      values: request.values,
      configuration: request.attentionConfiguration)
    return try FloatTensor(output.storage, shape: request.query.shape)
  }

  private static func expectError(
    name: String, failures: inout [JudgeFailure], operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(caseName: name, message: "expected an error"))
      return 0
    } catch {
      return 1
    }
  }
}
