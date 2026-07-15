import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P026RingKVCacheTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P026RingKVCacheJudge.evaluate(P026RingKVCacheSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.passedCaseCount, 3)
  }

  func testRingRejectsPositionGapAndOverwrittenRead() throws {
    let configuration = try KVCacheConfiguration(
      layerCount: 1, keyValueHeadCount: 1, headDimension: 1, capacity: 2)
    let cache = RingKVCache(configuration: configuration)
    let one = try FloatTensor([1], shape: [1, 1])
    try cache.append(layer: 0, logicalPosition: 5, key: one, value: one)
    XCTAssertThrowsError(
      try cache.append(layer: 0, logicalPosition: 7, key: one, value: one))
    try cache.append(layer: 0, logicalPosition: 6, key: one, value: one)
    try cache.append(layer: 0, logicalPosition: 7, key: one, value: one)
    XCTAssertEqual(try cache.logicalPositions(layer: 0), [6, 7])
    XCTAssertThrowsError(try cache.keyVector(layer: 0, logicalPosition: 5, head: 0))
  }

  func testJudgeRejectsUnboundedHistory() {
    let report = P026RingKVCacheJudge.evaluate { request in
      RingKVCacheResult(
        chronologicalHistory: [Array(request.firstLogicalPosition...request.queryLogicalPosition)],
        finalSnapshot: KVCacheLayerSnapshot(
          logicalPositions: Array(request.firstLogicalPosition...request.queryLogicalPosition),
          keys: request.keys,
          values: request.values),
        attentionOutput: try FloatTensor(
          Array(repeating: 0, count: request.query.elementCount), shape: request.query.shape),
        allocatedBytes: request.configuration.allocatedFloat32Bytes,
        storageAddressesStable: true)
    }
    XCTAssertFalse(report.isPassing)
  }
}
