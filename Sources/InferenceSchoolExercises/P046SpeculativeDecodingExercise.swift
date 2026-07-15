import InferenceSchoolCore

public enum P046SpeculativeDecodingExercise {
  public static func decodeBlock(
    prefix: [Int],
    maximumDraftTokens: Int,
    vocabularySize: Int,
    draft: TokenDistributionProvider,
    target: TokenDistributionProvider,
    generator: inout SeededGenerator
  ) throws -> SpeculativeBlockResult {
    try P046SpeculativeDecodingContract.validate(
      maximumDraftTokens: maximumDraftTokens,
      vocabularySize: vocabularySize)
    _ = try P046DistributionMath.normalized(
      draft(prefix), vocabularySize: vocabularySize, modelName: "draft")
    _ = try P046DistributionMath.normalized(
      target(prefix), vocabularySize: vocabularySize, modelName: "target")
    // TODO: draft K tokens, verify with min(1, p/q), correct rejection, and emit a bonus.
    return SpeculativeBlockResult(
      emittedTokenIDs: [],
      proposals: [],
      verifications: [],
      bonus: nil,
      rejectedAtProposalIndex: nil,
      draftEvaluationCount: 0,
      targetEvaluationCount: 0)
  }
}