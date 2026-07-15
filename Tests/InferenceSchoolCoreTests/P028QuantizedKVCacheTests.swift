import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P028QuantizedKVCacheTests: XCTestCase {
  func testCanonicalCPUPassesFloatOracleAndMetadataChecks() {
    let report = P028QuantizedKVCacheJudge.evaluate(P028QuantizedKVCacheSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.passedCaseCount, 3)
  }

  func testZeroVectorUsesFiniteScaleAndRoundTrips() throws {
    let configuration = try KVCacheConfiguration(
      layerCount: 1, keyValueHeadCount: 1, headDimension: 3, capacity: 1)
    let cache = QuantizedKVCache(configuration: configuration)
    let zero = try FloatTensor([0, 0, 0], shape: [1, 3])
    try cache.append(layer: 0, logicalPosition: 4, key: zero, value: zero)
    XCTAssertEqual(try cache.keyVector(layer: 0, logicalPosition: 4, head: 0), [0, 0, 0])
    XCTAssertEqual(cache.rawKeyScales(), [1])
    XCTAssertEqual(cache.allocatedBytes, 14)
  }

  func testJudgeRejectsFloatStorageAccounting() {
    let report = P028QuantizedKVCacheJudge.evaluate { request in
      QuantizedKVCacheResult(
        attentionOutput: try P023CachedAttentionSolution.run(request).output,
        dequantizedKeys: request.keys,
        dequantizedValues: request.values,
        keyScales: [],
        valueScales: [],
        allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes,
        maximumKeyError: 0,
        maximumValueError: 0)
    }
    XCTAssertFalse(report.isPassing)
  }
}
