import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P020TiledAttentionMetalTests: XCTestCase {
  func testCanonicalTiledMetalPassesJudge() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let p = try P020TiledAttentionSolution.makeMetalPipeline()
    let r = P020TiledAttentionJudge.evaluate { try p.apply($0, $1, $2, configuration: $3) }
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
  func testHeadDimensionLimitIsExplicit() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let p = try P020TiledAttentionSolution.makeMetalPipeline()
    let c = try AttentionConfiguration(queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 129)
    let t = try FloatTensor(Array(repeating: 1, count: 129), shape: [1, 1, 129])
    XCTAssertThrowsError(try p.apply(t, t, t, configuration: c))
  }
}
