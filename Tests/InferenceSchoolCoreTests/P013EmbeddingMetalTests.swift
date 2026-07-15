import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P013EmbeddingMetalTests: XCTestCase {
  func testCanonicalMetalSolutionPassesJudge() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let pipeline = try P013EmbeddingSolution.makeMetalPipeline()
    let report = P013EmbeddingJudge.evaluate(pipeline.apply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }
}
