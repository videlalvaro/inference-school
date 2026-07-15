import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P046SpeculativeDecodingTests: XCTestCase {
  func testCanonicalSpeculativeDecoderPassesJudge() {
    let report = P046SpeculativeDecodingJudge.evaluate(
      P046SpeculativeDecodingSolution.decodeBlock)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testAcceptAllTraceIsExactForSeededFixture() throws {
    let prefix = [7]
    var generator = SeededGenerator(seed: 123)
    let result = try P046SpeculativeDecodingSolution.decodeBlock(
      prefix: prefix,
      maximumDraftTokens: 2,
      vocabularySize: 2,
      draft: { _ in .probabilities([1, 0]) },
      target: { current in
        current.count < 3 ? .probabilities([1, 0]) : .probabilities([0, 1])
      },
      generator: &generator)
    XCTAssertEqual(result.emittedTokenIDs, [0, 0, 1])
    XCTAssertEqual(result.proposals.map(\.draftProbability), [1, 1])
    XCTAssertEqual(result.verifications.map(\.acceptanceRatio), [1, 1])
    XCTAssertEqual(result.targetEvaluationCount, 3)
    XCTAssertNotNil(result.bonus)
  }

  func testRejectFixtureUsesCorrectionAndStopsBlock() throws {
    var generator = SeededGenerator(seed: 456)
    let result = try P046SpeculativeDecodingSolution.decodeBlock(
      prefix: [],
      maximumDraftTokens: 3,
      vocabularySize: 2,
      draft: { _ in .probabilities([1, 0]) },
      target: { _ in .probabilities([0, 1]) },
      generator: &generator)
    XCTAssertEqual(result.proposals.count, 3)
    XCTAssertEqual(result.verifications.count, 1)
    XCTAssertEqual(result.emittedTokenIDs, [1])
    XCTAssertEqual(result.verifications[0].rejectionDistribution, [0, 1])
    XCTAssertEqual(result.rejectedAtProposalIndex, 0)
    XCTAssertNil(result.bonus)
  }

  func testOneStepEnumerableDistributionEqualsTarget() throws {
    let target = [0.1, 0.3, 0.6]
    let draft = [0.6, 0.3, 0.1]
    let output = try P046DistributionMath.oneStepOutputDistribution(
      target: target, draft: draft)
    XCTAssertEqual(output.count, target.count)
    for (actual, expected) in zip(output, target) {
      XCTAssertEqual(actual, expected, accuracy: 1e-12)
    }
  }

  func testTargetOnlyBaselineUsesOneEvaluationPerToken() throws {
    var generator = SeededGenerator(seed: 99)
    let result = try P046SpeculativeDecodingSolution.sampleTargetOnly(
      prefix: [],
      tokenCount: 3,
      vocabularySize: 2,
      target: { prefix in
        prefix.count.isMultiple(of: 2)
          ? .probabilities([1, 0])
          : .probabilities([0, 1])
      },
      generator: &generator)
    XCTAssertEqual(result.emittedTokenIDs, [0, 1, 0])
    XCTAssertEqual(result.targetEvaluationCount, 3)
    XCTAssertEqual(result.samplingDraws.count, 3)
  }

  func testJudgeRejectsTokenAgreementHeuristicWithoutProbabilityTrace() {
    let report = P046SpeculativeDecodingJudge.evaluate {
      prefix, maximumDraftTokens, vocabularySize, draft, target, generator in
      try P046SpeculativeDecodingContract.validate(
        maximumDraftTokens: maximumDraftTokens,
        vocabularySize: vocabularySize)
      _ = try draft(prefix)
      _ = try target(prefix)
      _ = generator.next()
      return SpeculativeBlockResult(
        emittedTokenIDs: [],
        proposals: [],
        verifications: [],
        bonus: nil,
        rejectedAtProposalIndex: nil,
        draftEvaluationCount: 1,
        targetEvaluationCount: 1)
    }
    XCTAssertFalse(report.isPassing)
  }
}