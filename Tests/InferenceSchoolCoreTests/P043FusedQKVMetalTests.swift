import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P043FusedQKVMetalTests: XCTestCase {
  func testCanonicalMetalKernelPassesJudge() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let pipeline = try P043FusedQKVSolution.makeMetalPipeline()
    let report = P043FusedQKVJudge.evaluate(pipeline.project)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }
}