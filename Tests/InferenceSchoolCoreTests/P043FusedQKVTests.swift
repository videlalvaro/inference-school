import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P043FusedQKVTests: XCTestCase {
  func testCanonicalFusedCPUPassesJudge() {
    let report = P043FusedQKVJudge.evaluate(P043FusedQKVSolution.fused)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testSeparateBaselinePassesSameJudge() {
    let report = P043FusedQKVJudge.evaluate(P043FusedQKVSolution.separate)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsProjectionWithoutNormalization() {
    let report = P043FusedQKVJudge.evaluate { request in
      try P043FusedQKVContract.validate(request)
      let sequence = request.input.shape[0]
      let headDimension = request.configuration.headDimension
      return FusedQKVResult(
        queries: try FloatTensor(
          Array(repeating: 0, count: sequence * request.configuration.queryProjectionDimension),
          shape: [sequence, request.configuration.queryHeadCount, headDimension]),
        keys: try FloatTensor(
          Array(repeating: 0, count: sequence * request.configuration.keyValueProjectionDimension),
          shape: [sequence, request.configuration.keyValueHeadCount, headDimension]),
        values: try FloatTensor(
          Array(repeating: 0, count: sequence * request.configuration.keyValueProjectionDimension),
          shape: [sequence, request.configuration.keyValueHeadCount, headDimension]))
    }
    XCTAssertFalse(report.isPassing)
  }

  func testCostModelRemovesNormalizedIntermediateAndThreeDispatches() throws {
    let configuration = try DecoderConfiguration(
      modelDimension: 4,
      hiddenDimension: 6,
      queryHeadCount: 2,
      keyValueHeadCount: 1,
      headDimension: 2,
      rotaryDimension: 2,
      rmsNormEpsilon: 1e-5)
    let request = FusedQKVRequest(
      input: try FloatTensor([1, 2, 3, 4, -1, 0.5, 2, -3], shape: [2, 4]),
      gamma: try FloatTensor([1, 1, 1, 1], shape: [4]),
      queryWeights: try FloatTensor(Array(repeating: 0.25, count: 16), shape: [4, 4]),
      keyWeights: try FloatTensor(Array(repeating: 0.5, count: 8), shape: [2, 4]),
      valueWeights: try FloatTensor(Array(repeating: -0.5, count: 8), shape: [2, 4]),
      epsilon: 1e-5,
      configuration: configuration)
    let comparison = try P043FusedQKVCostModel.compare(request)
    XCTAssertEqual(comparison.separate.dispatchCount, 4)
    XCTAssertEqual(comparison.fused.dispatchCount, 1)
    XCTAssertEqual(comparison.separate.intermediateBytes, 2 * 4 * MemoryLayout<Float>.stride)
    XCTAssertEqual(comparison.fused.intermediateBytes, 0)
    XCTAssertLessThan(comparison.fused.logicalTensorBytes, comparison.separate.logicalTensorBytes)
  }
}