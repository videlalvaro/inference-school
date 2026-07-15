import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P015RoPEMetalTests: XCTestCase {
  func testCanonicalMetalSolutionPassesJudge() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let pipeline = try P015RoPESolution.makeMetalPipeline()
    let report = P015RoPEJudge.evaluate {
      queries, keys, rotaryDimension, base, queryOffset, keyOffset in
      try pipeline.apply(
        queries, keys, rotaryDimension: rotaryDimension, base: base,
        queryPositionOffset: queryOffset, keyPositionOffset: keyOffset)
    }
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }
}
