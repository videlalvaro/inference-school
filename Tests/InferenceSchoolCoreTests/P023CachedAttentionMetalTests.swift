import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P023CachedAttentionMetalTests: XCTestCase {
  func testCanonicalMetalReadsCacheStorageAndPassesOracle() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let pipeline = try P023CachedAttentionSolution.makeMetalPipeline()
    let report = P023CachedAttentionJudge.evaluate {
      try P023CachedAttentionSolution.runMetal($0, pipeline: pipeline)
    }
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }
}
