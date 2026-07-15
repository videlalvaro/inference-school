import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P023CachedAttentionTests: XCTestCase {
  func testCanonicalCPUPassesMaterializedOracle() {
    let report = P023CachedAttentionJudge.evaluate(P023CachedAttentionSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.passedCaseCount, 4)
  }

  func testJudgeRejectsZeroAttention() {
    let report = P023CachedAttentionJudge.evaluate { request in
      CachedAttentionResult(
        output: try FloatTensor(
          Array(repeating: 0, count: request.query.elementCount), shape: request.query.shape),
        cachedLogicalPositions: Array(
          request.firstLogicalPosition...request.queryLogicalPosition),
        allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes)
    }
    XCTAssertFalse(report.isPassing)
  }
}
