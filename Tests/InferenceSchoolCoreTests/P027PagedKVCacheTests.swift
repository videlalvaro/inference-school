import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P027PagedKVCacheTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P027PagedKVCacheJudge.evaluate(P027PagedKVCacheSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.passedCaseCount, 4)
  }

  func testAllocatorRejectsExhaustionAndReusesFreedPage() throws {
    let configuration = try KVCacheConfiguration(
      layerCount: 2, keyValueHeadCount: 1, headDimension: 1, capacity: 2)
    let cache = try PagedKVCache(
      configuration: configuration, pageSize: 1, physicalPageCount: 1)
    let one = try FloatTensor([1], shape: [1, 1])
    try cache.append(layer: 0, logicalPosition: 0, key: one, value: one)
    XCTAssertThrowsError(
      try cache.append(layer: 1, logicalPosition: 0, key: one, value: one))
    try cache.free(layer: 0)
    try cache.append(layer: 1, logicalPosition: 4, key: one, value: one)
    XCTAssertEqual(try cache.physicalPages(layer: 1), [0])
  }

  func testJudgeRejectsContiguousPhysicalAssumption() {
    let report = P027PagedKVCacheJudge.evaluate { request in
      let empty = try (0..<request.configuration.layerCount).map { _ in
        try KVCacheLayerSnapshot(
          logicalPositions: [],
          keys: FloatTensor([], shape: [0, 1, 2]),
          values: FloatTensor([], shape: [0, 1, 2]))
      }
      return PagedKVCacheResult(
        layerSnapshots: empty,
        physicalPageTables: [[0, 1], [], [2]],
        allocatorReport: PageAllocatorReport(
          physicalPageCount: 3, allocatedPageCount: 3, freePageCount: 0,
          liveTokenCount: 0, internalFragmentSlots: 6, largestContiguousFreeRun: 0),
        attentionOutput: try FloatTensor([0, 0], shape: [1, 2]),
        allocatedBytes: 96)
    }
    XCTAssertFalse(report.isPassing)
  }
}
