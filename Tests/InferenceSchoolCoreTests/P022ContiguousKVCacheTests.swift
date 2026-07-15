import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P022ContiguousKVCacheTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P022ContiguousKVCacheJudge.evaluate(P022ContiguousKVCacheSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.passedCaseCount, 5)
  }

  func testCacheKeepsStorageStableAndLayersIsolated() throws {
    let configuration = try KVCacheConfiguration(
      layerCount: 2, keyValueHeadCount: 1, headDimension: 2, capacity: 2)
    let cache = ContiguousKVCache(configuration: configuration)
    let addresses = cache.storageAddresses()
    let key = try FloatTensor([1, 2], shape: [1, 2])
    let value = try FloatTensor([3, 4], shape: [1, 2])
    try cache.append(layer: 1, logicalPosition: 9, key: key, value: value)

    XCTAssertEqual(addresses.key, cache.storageAddresses().key)
    XCTAssertEqual(addresses.value, cache.storageAddresses().value)
    XCTAssertEqual(cache.keyStorageCount, 8)
    XCTAssertEqual(cache.valueStorageCount, 8)
    XCTAssertEqual(try cache.count(layer: 0), 0)
    XCTAssertEqual(try cache.count(layer: 1), 1)
    XCTAssertEqual(try cache.keyVector(layer: 1, logicalPosition: 9, head: 0), [1, 2])
    XCTAssertThrowsError(try cache.keyVector(layer: 0, logicalPosition: 9, head: 0))
  }

  func testJudgeRejectsGrowingPlaceholderTranscript() {
    let report = P022ContiguousKVCacheJudge.evaluate { configuration, _ in
      ContiguousKVCacheTranscript(
        layers: [],
        allocatedBytes: configuration.allocatedFloat32Bytes,
        keyStorageCount: configuration.elementsPerTensor,
        valueStorageCount: configuration.elementsPerTensor,
        storageAddressesStable: true)
    }
    XCTAssertFalse(report.isPassing)
  }
}