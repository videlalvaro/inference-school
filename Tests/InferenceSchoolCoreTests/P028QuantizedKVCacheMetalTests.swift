import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P028QuantizedKVCacheMetalTests: XCTestCase {
  func testCanonicalMetalDequantizesWhileAttending() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let pipeline = try P028QuantizedKVCacheSolution.makeMetalPipeline()
    let report = P028QuantizedKVCacheJudge.evaluate {
      try P028QuantizedKVCacheSolution.runMetal($0, pipeline: pipeline)
    }
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }
}