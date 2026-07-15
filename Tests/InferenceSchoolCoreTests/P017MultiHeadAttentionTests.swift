import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P017MultiHeadAttentionTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let r = P017MultiHeadAttentionJudge.evaluate(P017MultiHeadAttentionSolution.apply)
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
  func testJudgeRejectsZeroHeads() {
    let r = P017MultiHeadAttentionJudge.evaluate { q, k, v, c in
      _ = try P017MultiHeadAttentionContract.validate(q, k, v, c)
      return try FloatTensor(Array(repeating: 0, count: q.elementCount), shape: q.shape)
    }
    XCTAssertFalse(r.isPassing)
  }
}
