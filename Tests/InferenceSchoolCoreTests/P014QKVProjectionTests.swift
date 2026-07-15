import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P014QKVProjectionTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P014QKVProjectionJudge.evaluate(P014QKVProjectionSolution.project)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsShapeOnlyProjection() {
    let report = P014QKVProjectionJudge.evaluate {
      hidden, queryWeights, keyWeights, valueWeights, configuration in
      try QKVProjectionContract.validate(
        hidden: hidden,
        queryWeights: queryWeights,
        keyWeights: keyWeights,
        valueWeights: valueWeights,
        configuration: configuration
      )
      let sequence = hidden.shape[0]
      return QKVProjectionResult(
        queries: try FloatTensor(
          Array(
            repeating: 0,
            count: sequence * configuration.queryHeadCount * configuration.headDimension),
          shape: [sequence, configuration.queryHeadCount, configuration.headDimension]),
        keys: try FloatTensor(
          Array(
            repeating: 0,
            count: sequence * configuration.keyValueHeadCount * configuration.headDimension),
          shape: [sequence, configuration.keyValueHeadCount, configuration.headDimension]),
        values: try FloatTensor(
          Array(
            repeating: 0,
            count: sequence * configuration.keyValueHeadCount * configuration.headDimension),
          shape: [sequence, configuration.keyValueHeadCount, configuration.headDimension])
      )
    }
    XCTAssertFalse(report.isPassing)
  }

  func testHeadReshapeIsAContiguousView() throws {
    let projected = try FloatTensor((0..<24).map(Float.init), shape: [2, 12])
    let view = try projected.view.reshaped(to: [2, 3, 4])
    XCTAssertEqual(try view.value(at: [1, 2, 3]), 23)
    XCTAssertEqual(view.strides, [12, 4, 1])
  }
}
