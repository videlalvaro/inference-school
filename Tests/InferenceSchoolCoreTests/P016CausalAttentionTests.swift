import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P016CausalAttentionTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P016CausalAttentionJudge.evaluate(P016CausalAttentionSolution.apply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }
  func testJudgeRejectsUniformAveraging() {
    let report = P016CausalAttentionJudge.evaluate { queries, keys, values, configuration in
      _ = try P016CausalAttentionContract.validate(
        queries: queries, keys: keys, values: values, configuration: configuration)
      return try FloatTensor(Array(repeating: 0, count: queries.elementCount), shape: queries.shape)
    }
    XCTAssertFalse(report.isPassing)
  }
  func testFirstRowCannotSeeFutureValues() throws {
    let configuration = try AttentionConfiguration(
      queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 1)
    let queries = try FloatTensor([1, 1], shape: [2, 1, 1])
    let keys = try FloatTensor([1, 1], shape: [2, 1, 1])
    let output = try P016CausalAttentionSolution.apply(
      queries, keys, FloatTensor([3, 999], shape: [2, 1, 1]), configuration)
    XCTAssertEqual(output.storage[0], 3)
  }
}
