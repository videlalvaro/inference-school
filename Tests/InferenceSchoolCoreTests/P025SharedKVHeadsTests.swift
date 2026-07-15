import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P025SharedKVHeadsTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P025SharedKVHeadsJudge.evaluate(P025SharedKVHeadsSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.passedCaseCount, 3)
  }

  func testModuloMappingFailsAndInvalidDivisibilityThrows() {
    let report = P025SharedKVHeadsJudge.evaluate(P025SharedKVHeadsExerciseLikeModulo.run)
    XCTAssertFalse(report.isPassing)
    XCTAssertThrowsError(
      try AttentionConfiguration(queryHeadCount: 3, keyValueHeadCount: 2, headDimension: 4))
  }
}

private enum P025SharedKVHeadsExerciseLikeModulo {
  static func run(_ request: CachedAttentionRequest) throws -> SharedKVHeadsResult {
    let c = request.attentionConfiguration
    var output = Array(repeating: Float.zero, count: request.query.elementCount)
    for queryHead in 0..<c.queryHeadCount {
      let wrongHead = queryHead % c.keyValueHeadCount
      for feature in 0..<c.headDimension {
        output[queryHead * c.headDimension + feature]
          = request.values.storage[((request.tokenCount - 1) * c.keyValueHeadCount + wrongHead)
            * c.headDimension + feature]
      }
    }
    return SharedKVHeadsResult(
      attention: CachedAttentionResult(
        output: try FloatTensor(output, shape: request.query.shape),
        cachedLogicalPositions: Array(
          request.firstLogicalPosition...request.queryLogicalPosition),
        allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes),
      bytes: try KVHeadMemoryModel.compare(
        layerCount: request.cacheConfiguration.layerCount,
        tokenCount: request.cacheConfiguration.capacity,
        queryHeadCount: c.queryHeadCount,
        gqaHeadCount: c.keyValueHeadCount,
        headDimension: c.headDimension))
  }
}