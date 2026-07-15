import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P021SlidingWindowAttentionMetalTests: XCTestCase {
  func testCanonicalMetalSolutionPassesJudge() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let p = try P021SlidingWindowAttentionSolution.makeMetalPipeline()
    let r = P021SlidingWindowAttentionJudge.evaluate {
      try p.apply($0, $1, $2, configuration: $3, window: $4)
    }
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
}
