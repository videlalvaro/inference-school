import InferenceSchoolCore

public enum P046SpeculativeDecodingSolution {
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

    var proposalPrefix = prefix
    var proposals: [DraftProposalTrace] = []
    for index in 0..<maximumDraftTokens {
      let distribution = try P046DistributionMath.normalized(
        draft(proposalPrefix),
        vocabularySize: vocabularySize,
        modelName: "draft")
      let sampled = sample(distribution, generator: &generator)
      let probability = distribution[sampled.tokenID]
      guard probability > 0 else {
        throw SpeculativeDecodingError.sampledZeroDraftProbability(tokenID: sampled.tokenID)
      }
      proposals.append(DraftProposalTrace(
        index: index,
        prefix: proposalPrefix,
        tokenID: sampled.tokenID,
        draftDistribution: distribution,
        draftProbability: probability,
        samplingDraw: sampled.draw))
      proposalPrefix.append(sampled.tokenID)
    }

    var emitted: [Int] = []
    var verifications: [VerificationTrace] = []
    for proposal in proposals {
      let targetDistribution = try P046DistributionMath.normalized(
        target(proposal.prefix),
        vocabularySize: vocabularySize,
        modelName: "target")
      let targetProbability = targetDistribution[proposal.tokenID]
      let ratio = min(1, targetProbability / proposal.draftProbability)
      let acceptanceDraw = generator.nextUnitInterval()
      if acceptanceDraw < ratio {
        emitted.append(proposal.tokenID)
        verifications.append(VerificationTrace(
          proposalIndex: proposal.index,
          tokenID: proposal.tokenID,
          targetDistribution: targetDistribution,
          targetProbability: targetProbability,
          draftProbability: proposal.draftProbability,
          acceptanceRatio: ratio,
          acceptanceDraw: acceptanceDraw,
          accepted: true,
          rejectionDistribution: nil,
          replacementTokenID: nil,
          replacementSamplingDraw: nil))
      } else {
        let correction = try P046DistributionMath.correction(
          target: targetDistribution,
          draft: proposal.draftDistribution)
        let replacement = sample(correction, generator: &generator)
        emitted.append(replacement.tokenID)
        verifications.append(VerificationTrace(
          proposalIndex: proposal.index,
          tokenID: proposal.tokenID,
          targetDistribution: targetDistribution,
          targetProbability: targetProbability,
          draftProbability: proposal.draftProbability,
          acceptanceRatio: ratio,
          acceptanceDraw: acceptanceDraw,
          accepted: false,
          rejectionDistribution: correction,
          replacementTokenID: replacement.tokenID,
          replacementSamplingDraw: replacement.draw))
        return SpeculativeBlockResult(
          emittedTokenIDs: emitted,
          proposals: proposals,
          verifications: verifications,
          bonus: nil,
          rejectedAtProposalIndex: proposal.index,
          draftEvaluationCount: proposals.count,
          targetEvaluationCount: verifications.count)
      }
    }

    let bonusPrefix = prefix + proposals.map(\.tokenID)
    let bonusDistribution = try P046DistributionMath.normalized(
      target(bonusPrefix),
      vocabularySize: vocabularySize,
      modelName: "target")
    let bonusSample = sample(bonusDistribution, generator: &generator)
    emitted.append(bonusSample.tokenID)
    return SpeculativeBlockResult(
      emittedTokenIDs: emitted,
      proposals: proposals,
      verifications: verifications,
      bonus: BonusTokenTrace(
        prefix: bonusPrefix,
        targetDistribution: bonusDistribution,
        tokenID: bonusSample.tokenID,
        samplingDraw: bonusSample.draw),
      rejectedAtProposalIndex: nil,
      draftEvaluationCount: proposals.count,
      targetEvaluationCount: verifications.count + 1)
  }

  public static func sampleTargetOnly(
    prefix: [Int],
    tokenCount: Int,
    vocabularySize: Int,
    target: TokenDistributionProvider,
    generator: inout SeededGenerator
  ) throws -> TargetOnlySamplingResult {
    guard tokenCount >= 0 else {
      throw SpeculativeDecodingError.invalidTargetOnlyTokenCount(tokenCount)
    }
    guard vocabularySize > 0 else {
      throw SpeculativeDecodingError.invalidVocabularySize(vocabularySize)
    }
    var current = prefix
    var tokens: [Int] = []
    var draws: [Double] = []
    for _ in 0..<tokenCount {
      let distribution = try P046DistributionMath.normalized(
        target(current),
        vocabularySize: vocabularySize,
        modelName: "target")
      let sampled = sample(distribution, generator: &generator)
      tokens.append(sampled.tokenID)
      draws.append(sampled.draw)
      current.append(sampled.tokenID)
    }
    return TargetOnlySamplingResult(
      emittedTokenIDs: tokens,
      targetEvaluationCount: tokenCount,
      samplingDraws: draws)
  }

  private static func sample(
    _ probabilities: [Double],
    generator: inout SeededGenerator
  ) -> (tokenID: Int, draw: Double) {
    let draw = generator.nextUnitInterval()
    var cumulative = 0.0
    for (tokenID, probability) in probabilities.enumerated() {
      cumulative += probability
      if draw < cumulative { return (tokenID, draw) }
    }
    return (probabilities.count - 1, draw)
  }
}