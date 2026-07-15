import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P016CausalAttentionMetalTests: XCTestCase {
  func testCanonicalMetalSolutionPassesJudge() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let pipeline = try P016CausalAttentionSolution.makeMetalPipeline()
    let report = P016CausalAttentionJudge.evaluate {
      try pipeline.apply($0, $1, $2, configuration: $3)
    }
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }
}
