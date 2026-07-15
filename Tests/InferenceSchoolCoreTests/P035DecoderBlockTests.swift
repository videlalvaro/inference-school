import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P035DecoderBlockTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P035DecoderBlockJudge.evaluate(P035DecoderBlockSolution.apply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsMissingRoPE() {
    let report = P035DecoderBlockJudge.evaluate { state, weights, configuration in
      let result = try P035DecoderBlockSolution.apply(
        state: state, weights: weights, configuration: configuration)
      let captured = result.intermediates
      return DecoderBlockResult(
        state: result.state,
        intermediates: DecoderBlockIntermediates(
          attentionNormalized: captured.attentionNormalized,
          queries: captured.queries,
          keys: captured.keys,
          values: captured.values,
          rotatedQueries: captured.queries,
          rotatedKeys: captured.keys,
          attentionHeads: captured.attentionHeads,
          concatenatedAttention: captured.concatenatedAttention,
          attentionProjection: captured.attentionProjection,
          postAttentionResidual: captured.postAttentionResidual,
          mlpNormalized: captured.mlpNormalized,
          gateProjection: captured.gateProjection,
          upProjection: captured.upProjection,
          activatedGate: captured.activatedGate,
          gatedHidden: captured.gatedHidden,
          downProjection: captured.downProjection))
    }
    XCTAssertFalse(report.isPassing)
  }

  func testJudgeRejectsNormalizationBeforeWrongResidual() {
    let report = P035DecoderBlockJudge.evaluate { state, weights, configuration in
      let result = try P035DecoderBlockSolution.apply(
        state: state, weights: weights, configuration: configuration)
      let captured = result.intermediates
      return DecoderBlockResult(
        state: result.state,
        intermediates: DecoderBlockIntermediates(
          attentionNormalized: captured.attentionNormalized,
          queries: captured.queries,
          keys: captured.keys,
          values: captured.values,
          rotatedQueries: captured.rotatedQueries,
          rotatedKeys: captured.rotatedKeys,
          attentionHeads: captured.attentionHeads,
          concatenatedAttention: captured.concatenatedAttention,
          attentionProjection: captured.attentionProjection,
          postAttentionResidual: captured.postAttentionResidual,
          mlpNormalized: captured.attentionNormalized,
          gateProjection: captured.gateProjection,
          upProjection: captured.upProjection,
          activatedGate: captured.activatedGate,
          gatedHidden: captured.gatedHidden,
          downProjection: captured.downProjection))
    }
    XCTAssertFalse(report.isPassing)
  }

  func testConfigurationRejectsInvalidHeadAndRotaryContracts() throws {
    XCTAssertThrowsError(try DecoderConfiguration(
      modelDimension: 6,
      hiddenDimension: 8,
      queryHeadCount: 3,
      keyValueHeadCount: 2,
      headDimension: 2,
      rotaryDimension: 2,
      rmsNormEpsilon: 1e-5))
    XCTAssertThrowsError(try DecoderConfiguration(
      modelDimension: 4,
      hiddenDimension: 8,
      queryHeadCount: 2,
      keyValueHeadCount: 1,
      headDimension: 2,
      rotaryDimension: 3,
      rmsNormEpsilon: 1e-5))
  }
}