import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P019OnlineAttentionMetalTests: XCTestCase {
  func testCanonicalMetalSolutionPassesMaterializedOracle() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let p = try P019OnlineAttentionSolution.makeMetalPipeline()
    let r = P019OnlineAttentionJudge.evaluate { try p.apply($0, $1, $2, configuration: $3) }
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
}
