import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P019OnlineAttentionTests: XCTestCase {
  func testCanonicalSolutionPassesMaterializedOracle() {
    let r = P019OnlineAttentionJudge.evaluate(P019OnlineAttentionSolution.apply)
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
  func testJudgeRejectsLastValueOnly() {
    let r = P019OnlineAttentionJudge.evaluate { q, k, v, c in
      let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
      try validateVisibleKeys(input)
      return try FloatTensor(
        Array(repeating: v.storage.last ?? 0, count: q.elementCount), shape: q.shape)
    }
    XCTAssertFalse(r.isPassing)
  }
}
