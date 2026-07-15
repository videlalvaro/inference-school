import Foundation

public enum PagedKVOperation: Sendable, Equatable {
  case append(layer: Int, logicalPosition: Int, key: FloatTensor, value: FloatTensor)
  case free(layer: Int)
}

public struct PagedKVCacheRequest: Sendable, Equatable {
  public let configuration: KVCacheConfiguration
  public let pageSize: Int
  public let physicalPageCount: Int
  public let operations: [PagedKVOperation]
  public let attentionLayer: Int
  public let queryLogicalPosition: Int
  public let query: FloatTensor
  public let queryHeadCount: Int

  public init(
    configuration: KVCacheConfiguration,
    pageSize: Int,
    physicalPageCount: Int,
    operations: [PagedKVOperation],
    attentionLayer: Int,
    queryLogicalPosition: Int,
    query: FloatTensor,
    queryHeadCount: Int
  ) throws {
    guard pageSize > 0 else { throw KVCacheError.invalidPageSize(pageSize) }
    guard physicalPageCount > 0 else { throw KVCacheError.invalidPageCount(physicalPageCount) }
    try configuration.validate(layer: attentionLayer)
    guard queryLogicalPosition >= 0 else {
      throw KVCacheError.invalidLogicalPosition(queryLogicalPosition)
    }
    _ = try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: configuration.keyValueHeadCount,
      headDimension: configuration.headDimension)
    let expectedQuery = [queryHeadCount, configuration.headDimension]
    guard query.shape == expectedQuery else {
      throw KVCacheError.vectorShapeMismatch(
        name: "Query", expected: expectedQuery, actual: query.shape)
    }
    self.configuration = configuration
    self.pageSize = pageSize
    self.physicalPageCount = physicalPageCount
    self.operations = operations
    self.attentionLayer = attentionLayer
    self.queryLogicalPosition = queryLogicalPosition
    self.query = query
    self.queryHeadCount = queryHeadCount
  }
}

public struct PageAllocatorReport: Sendable, Equatable {
  public let physicalPageCount: Int
  public let allocatedPageCount: Int
  public let freePageCount: Int
  public let liveTokenCount: Int
  public let internalFragmentSlots: Int
  public let largestContiguousFreeRun: Int

  public init(
    physicalPageCount: Int,
    allocatedPageCount: Int,
    freePageCount: Int,
    liveTokenCount: Int,
    internalFragmentSlots: Int,
    largestContiguousFreeRun: Int
  ) {
    self.physicalPageCount = physicalPageCount
    self.allocatedPageCount = allocatedPageCount
    self.freePageCount = freePageCount
    self.liveTokenCount = liveTokenCount
    self.internalFragmentSlots = internalFragmentSlots
    self.largestContiguousFreeRun = largestContiguousFreeRun
  }
}

public struct PagedKVCacheResult: Sendable, Equatable {
  public let layerSnapshots: [KVCacheLayerSnapshot]
  public let physicalPageTables: [[Int]]
  public let allocatorReport: PageAllocatorReport
  public let attentionOutput: FloatTensor
  public let allocatedBytes: Int

  public init(
    layerSnapshots: [KVCacheLayerSnapshot],
    physicalPageTables: [[Int]],
    allocatorReport: PageAllocatorReport,
    attentionOutput: FloatTensor,
    allocatedBytes: Int
  ) {
    self.layerSnapshots = layerSnapshots
    self.physicalPageTables = physicalPageTables
    self.allocatorReport = allocatorReport
    self.attentionOutput = attentionOutput
    self.allocatedBytes = allocatedBytes
  }
}

public typealias PagedKVCacheImplementation = (
  _ request: PagedKVCacheRequest
) throws -> PagedKVCacheResult

public enum P027PagedKVCacheJudge {
  public static func evaluate(_ implementation: PagedKVCacheImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let configuration = try KVCacheConfiguration(
        layerCount: 3, keyValueHeadCount: 1, headDimension: 2, capacity: 4)
      func vector(_ a: Float, _ b: Float) throws -> FloatTensor {
        try FloatTensor([a, b], shape: [1, 2])
      }
      let operations: [PagedKVOperation] = [
        .append(layer: 0, logicalPosition: 10, key: try vector(1, 0), value: try vector(10, 11)),
        .append(layer: 0, logicalPosition: 11, key: try vector(0, 1), value: try vector(12, 13)),
        .append(layer: 1, logicalPosition: 20, key: try vector(2, 0), value: try vector(20, 21)),
        .append(layer: 1, logicalPosition: 21, key: try vector(0, 2), value: try vector(22, 23)),
        .append(layer: 0, logicalPosition: 12, key: try vector(1, 1), value: try vector(14, 15)),
        .free(layer: 1),
        .append(layer: 2, logicalPosition: 30, key: try vector(3, 0), value: try vector(30, 31)),
        .append(layer: 2, logicalPosition: 31, key: try vector(0, 3), value: try vector(32, 33)),
        .append(layer: 0, logicalPosition: 13, key: try vector(-1, 1), value: try vector(16, 17)),
      ]
      let request = try PagedKVCacheRequest(
        configuration: configuration,
        pageSize: 2,
        physicalPageCount: 3,
        operations: operations,
        attentionLayer: 0,
        queryLogicalPosition: 13,
        query: FloatTensor([0.5, 1], shape: [1, 2]),
        queryHeadCount: 1)
      let actual = try implementation(request)
      if actual.physicalPageTables == [[0, 2], [], [1]] {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "free reuse and noncontiguous page table",
          message: "expected layer page tables [[0,2],[],[1]]"))
      }
      if actual.layerSnapshots[0].logicalPositions == [10, 11, 12, 13],
        actual.layerSnapshots[0].values.storage == [10, 11, 12, 13, 14, 15, 16, 17],
        actual.layerSnapshots[2].logicalPositions == [30, 31],
        actual.layerSnapshots[2].values.storage == [30, 31, 32, 33]
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "gather crosses page boundary and physical order",
          message: "logical reads assumed physical contiguity or retained freed values"))
      }
      let expectedReport = PageAllocatorReport(
        physicalPageCount: 3,
        allocatedPageCount: 3,
        freePageCount: 0,
        liveTokenCount: 6,
        internalFragmentSlots: 0,
        largestContiguousFreeRun: 0)
      if actual.allocatorReport == expectedReport,
        actual.allocatedBytes == 3 * 2 * 1 * 2 * 2 * MemoryLayout<Float>.stride
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "deterministic page accounting",
          message: "allocator page, fragmentation, or byte accounting differs"))
      }
      let keys = try FloatTensor([1, 0, 0, 1, 1, 1, -1, 1], shape: [4, 1, 2])
      let values = try FloatTensor([10, 11, 12, 13, 14, 15, 16, 17], shape: [4, 1, 2])
      let attention = try AttentionConfiguration(
        queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2,
        queryPositionOffset: 13, keyPositionOffset: 10)
      let query = try FloatTensor(request.query.storage, shape: [1, 1, 2])
      let expected3D = try AttentionJudgeOracle.materialized(
        queries: query, keys: keys, values: values, configuration: attention)
      let expected = try FloatTensor(expected3D.storage, shape: [1, 2])
      if AttentionJudgeOracle.approximatelyEqual(actual.attentionOutput, expected) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "paged gather feeds cached attention",
          message: "attention did not follow the logical page table"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "paged cache", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }
}
