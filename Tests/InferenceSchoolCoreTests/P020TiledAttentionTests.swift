import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P020TiledAttentionTests: XCTestCase {
  func testCanonicalCPUPathPassesJudge() {
    let r = P020TiledAttentionJudge.evaluate(P020TiledAttentionSolution.apply)
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
  func testJudgeRejectsZeroOutput() {
    let r = P020TiledAttentionJudge.evaluate { q, k, v, c in
      let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
      try validateVisibleKeys(input)
      return try FloatTensor(Array(repeating: 0, count: q.elementCount), shape: q.shape)
    }
    XCTAssertFalse(r.isPassing)
  }
}
