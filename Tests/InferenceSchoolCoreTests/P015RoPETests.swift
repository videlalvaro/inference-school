import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P015RoPETests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P015RoPEJudge.evaluate(P015RoPESolution.apply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsNoOpRotation() {
    let report = P015RoPEJudge.evaluate {
      queries, keys, rotaryDimension, base, queryOffset, keyOffset in
      try RoPEContract.validate(
        queries: queries, keys: keys, rotaryDimension: rotaryDimension, base: base,
        queryPositionOffset: queryOffset, keyPositionOffset: keyOffset)
      return RoPEResult(queries: queries, keys: keys)
    }
    XCTAssertFalse(report.isPassing)
  }

  func testPartialRotationPreservesSuffix() throws {
    let tensor = try FloatTensor([1, 2, 3, 4, 5, 6], shape: [1, 1, 6])
    let result = try P015RoPESolution.apply(
      queries: tensor, keys: tensor, rotaryDimension: 4, base: 10_000, queryPositionOffset: 2,
      keyPositionOffset: 2)
    XCTAssertEqual(Array(result.queries.storage[4...]), [5, 6])
  }
}
