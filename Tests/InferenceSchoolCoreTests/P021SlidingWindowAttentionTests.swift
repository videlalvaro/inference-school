import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P021SlidingWindowAttentionTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let r = P021SlidingWindowAttentionJudge.evaluate(P021SlidingWindowAttentionSolution.apply)
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
  func testJudgeRejectsFullCausalForEveryWindow() {
    let r = P021SlidingWindowAttentionJudge.evaluate { q, k, v, c, window in
      guard window > 0 else { throw AttentionError.invalidWindow(window) }
      return try P019OnlineAttentionSolution.apply(q, k, v, c)
    }
    XCTAssertFalse(r.isPassing)
  }
}
