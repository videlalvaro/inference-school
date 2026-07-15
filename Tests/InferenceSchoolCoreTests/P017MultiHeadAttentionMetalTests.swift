import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P017MultiHeadAttentionMetalTests: XCTestCase {
  func testCanonicalMetalSolutionPassesJudge() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let p = try P017MultiHeadAttentionSolution.makeMetalPipeline()
    let r = P017MultiHeadAttentionJudge.evaluate { try p.apply($0, $1, $2, configuration: $3) }
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
}
