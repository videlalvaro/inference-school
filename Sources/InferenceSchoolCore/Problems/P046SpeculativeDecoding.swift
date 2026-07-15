import Foundation

public enum SpeculativeDistribution: Sendable, Equatable {
  case logits([Double])
  case probabilities([Double])
}

public typealias TokenDistributionProvider = ([Int]) throws -> SpeculativeDistribution

public enum SpeculativeDecodingError: Error, Equatable, LocalizedError {
  case invalidVocabularySize(Int)
  case invalidDraftLength(Int)
  case invalidTargetOnlyTokenCount(Int)
  case vocabularyMismatch(model: String, expected: Int, actual: Int)
  case nonFiniteLogit(model: String, tokenID: Int)
  case invalidProbability(model: String, tokenID: Int, value: Double)
  case zeroProbabilityMass(model: String)
  case sampledZeroDraftProbability(tokenID: Int)
  case emptyCorrectionDistribution
  case invalidAcceptanceProbability(Double)
  case invalidCost(name: String, value: Double)

  public var errorDescription: String? {
    switch self {
    case let .invalidVocabularySize(value):
      "Vocabulary size must be positive; received \(value)."
    case let .invalidDraftLength(value):
      "Maximum draft length K must be positive; received \(value)."
    case let .invalidTargetOnlyTokenCount(value):
      "Target-only token count must be nonnegative; received \(value)."
    case let .vocabularyMismatch(model, expected, actual):
      "\(model) distribution must contain \(expected) entries; received \(actual)."
    case let .nonFiniteLogit(model, tokenID):
      "\(model) produced a non-finite logit for token \(tokenID)."
    case let .invalidProbability(model, tokenID, value):
      "\(model) produced invalid probability \(value) for token \(tokenID)."
    case let .zeroProbabilityMass(model):
      "\(model) distribution has zero probability mass."
    case let .sampledZeroDraftProbability(tokenID):
      "Draft sampling selected token \(tokenID) with zero probability."
    case .emptyCorrectionDistribution:
      "A rejected proposal requires positive mass in max(p - q, 0)."
    case let .invalidAcceptanceProbability(value):
      "Acceptance probability must be in [0, 1]; received \(value)."
    case let .invalidCost(name, value):
      "Cost \(name) must be finite and nonnegative; received \(value)."
    }
  }
}

public struct DraftProposalTrace: Sendable, Equatable {
  public let index: Int
  public let prefix: [Int]
  public let tokenID: Int
  public let draftDistribution: [Double]
  public let draftProbability: Double
  public let samplingDraw: Double

  public init(
    index: Int,
    prefix: [Int],
    tokenID: Int,
    draftDistribution: [Double],
    draftProbability: Double,
    samplingDraw: Double
  ) {
    self.index = index
    self.prefix = prefix
    self.tokenID = tokenID
    self.draftDistribution = draftDistribution
    self.draftProbability = draftProbability
    self.samplingDraw = samplingDraw
  }
}

public struct VerificationTrace: Sendable, Equatable {
  public let proposalIndex: Int
  public let tokenID: Int
  public let targetDistribution: [Double]
  public let targetProbability: Double
  public let draftProbability: Double
  public let acceptanceRatio: Double
  public let acceptanceDraw: Double
  public let accepted: Bool
  public let rejectionDistribution: [Double]?
  public let replacementTokenID: Int?
  public let replacementSamplingDraw: Double?

  public init(
    proposalIndex: Int,
    tokenID: Int,
    targetDistribution: [Double],
    targetProbability: Double,
    draftProbability: Double,
    acceptanceRatio: Double,
    acceptanceDraw: Double,
    accepted: Bool,
    rejectionDistribution: [Double]?,
    replacementTokenID: Int?,
    replacementSamplingDraw: Double?
  ) {
    self.proposalIndex = proposalIndex
    self.tokenID = tokenID
    self.targetDistribution = targetDistribution
    self.targetProbability = targetProbability
    self.draftProbability = draftProbability
    self.acceptanceRatio = acceptanceRatio
    self.acceptanceDraw = acceptanceDraw
    self.accepted = accepted
    self.rejectionDistribution = rejectionDistribution
    self.replacementTokenID = replacementTokenID
    self.replacementSamplingDraw = replacementSamplingDraw
  }
}

public struct BonusTokenTrace: Sendable, Equatable {
  public let prefix: [Int]
  public let targetDistribution: [Double]
  public let tokenID: Int
  public let samplingDraw: Double

  public init(
    prefix: [Int],
    targetDistribution: [Double],
    tokenID: Int,
    samplingDraw: Double
  ) {
    self.prefix = prefix
    self.targetDistribution = targetDistribution
    self.tokenID = tokenID
    self.samplingDraw = samplingDraw
  }
}

public struct SpeculativeBlockResult: Sendable, Equatable {
  public let emittedTokenIDs: [Int]
  public let proposals: [DraftProposalTrace]
  public let verifications: [VerificationTrace]
  public let bonus: BonusTokenTrace?
  public let rejectedAtProposalIndex: Int?
  public let draftEvaluationCount: Int
  public let targetEvaluationCount: Int

  public init(
    emittedTokenIDs: [Int],
    proposals: [DraftProposalTrace],
    verifications: [VerificationTrace],
    bonus: BonusTokenTrace?,
    rejectedAtProposalIndex: Int?,
    draftEvaluationCount: Int,
    targetEvaluationCount: Int
  ) {
    self.emittedTokenIDs = emittedTokenIDs
    self.proposals = proposals
    self.verifications = verifications
    self.bonus = bonus
    self.rejectedAtProposalIndex = rejectedAtProposalIndex
    self.draftEvaluationCount = draftEvaluationCount
    self.targetEvaluationCount = targetEvaluationCount
  }
}

public struct TargetOnlySamplingResult: Sendable, Equatable {
  public let emittedTokenIDs: [Int]
  public let targetEvaluationCount: Int
  public let samplingDraws: [Double]

  public init(
    emittedTokenIDs: [Int],
    targetEvaluationCount: Int,
    samplingDraws: [Double]
  ) {
    self.emittedTokenIDs = emittedTokenIDs
    self.targetEvaluationCount = targetEvaluationCount
    self.samplingDraws = samplingDraws
  }
}

public struct SpeculativeCostEstimate: Sendable, Equatable {
  public let expectedEmittedTokens: Double
  public let speculativeCost: Double
  public let targetOnlyCostForExpectedTokens: Double
  public let modeledSpeedup: Double

  public init(
    expectedEmittedTokens: Double,
    speculativeCost: Double,
    targetOnlyCostForExpectedTokens: Double,
    modeledSpeedup: Double
  ) {
    self.expectedEmittedTokens = expectedEmittedTokens
    self.speculativeCost = speculativeCost
    self.targetOnlyCostForExpectedTokens = targetOnlyCostForExpectedTokens
    self.modeledSpeedup = modeledSpeedup
  }
}

public typealias SpeculativeDecodingImplementation = (
  _ prefix: [Int],
  _ maximumDraftTokens: Int,
  _ vocabularySize: Int,
  _ draft: TokenDistributionProvider,
  _ target: TokenDistributionProvider,
  _ generator: inout SeededGenerator
) throws -> SpeculativeBlockResult

public enum P046DistributionMath {
  public static func normalized(
    _ distribution: SpeculativeDistribution,
    vocabularySize: Int,
    modelName: String
  ) throws -> [Double] {
    guard vocabularySize > 0 else {
      throw SpeculativeDecodingError.invalidVocabularySize(vocabularySize)
    }
    switch distribution {
    case .logits(let logits):
      guard logits.count == vocabularySize else {
        throw SpeculativeDecodingError.vocabularyMismatch(
          model: modelName, expected: vocabularySize, actual: logits.count)
      }
      for (tokenID, logit) in logits.enumerated() where !logit.isFinite {
        throw SpeculativeDecodingError.nonFiniteLogit(model: modelName, tokenID: tokenID)
      }
      let maximum = logits.max()!
      let exponentials = logits.map { exp($0 - maximum) }
      let sum = exponentials.reduce(0, +)
      guard sum.isFinite, sum > 0 else {
        throw SpeculativeDecodingError.zeroProbabilityMass(model: modelName)
      }
      return exponentials.map { $0 / sum }
    case .probabilities(let probabilities):
      guard probabilities.count == vocabularySize else {
        throw SpeculativeDecodingError.vocabularyMismatch(
          model: modelName, expected: vocabularySize, actual: probabilities.count)
      }
      for (tokenID, probability) in probabilities.enumerated()
        where !probability.isFinite || probability < 0
      {
        throw SpeculativeDecodingError.invalidProbability(
          model: modelName, tokenID: tokenID, value: probability)
      }
      let sum = probabilities.reduce(0, +)
      guard sum.isFinite, sum > 0 else {
        throw SpeculativeDecodingError.zeroProbabilityMass(model: modelName)
      }
      return probabilities.map { $0 / sum }
    }
  }

  public static func correction(target: [Double], draft: [Double]) throws -> [Double] {
    guard target.count == draft.count else {
      throw SpeculativeDecodingError.vocabularyMismatch(
        model: "correction", expected: target.count, actual: draft.count)
    }
    let positiveDifference = zip(target, draft).map { max($0 - $1, 0) }
    let sum = positiveDifference.reduce(0, +)
    guard sum.isFinite, sum > 0 else {
      throw SpeculativeDecodingError.emptyCorrectionDistribution
    }
    return positiveDifference.map { $0 / sum }
  }

  public static func oneStepOutputDistribution(
    target: [Double],
    draft: [Double]
  ) throws -> [Double] {
    guard target.count == draft.count else {
      throw SpeculativeDecodingError.vocabularyMismatch(
        model: "one-step proof", expected: target.count, actual: draft.count)
    }
    let acceptedMass = zip(target, draft).map { min($0, $1) }
    let rejectionMass = 1 - acceptedMass.reduce(0, +)
    if rejectionMass <= 1e-15 { return acceptedMass }
    let corrected = try correction(target: target, draft: draft)
    return zip(acceptedMass, corrected).map { $0 + rejectionMass * $1 }
  }
}

public enum P046SpeculativeCostModel {
  public static func estimate(
    maximumDraftTokens: Int,
    acceptanceProbability: Double,
    draftCostPerToken: Double,
    targetVerificationCost: Double,
    targetCostPerToken: Double
  ) throws -> SpeculativeCostEstimate {
    guard maximumDraftTokens > 0 else {
      throw SpeculativeDecodingError.invalidDraftLength(maximumDraftTokens)
    }
    guard acceptanceProbability.isFinite,
      acceptanceProbability >= 0,
      acceptanceProbability <= 1
    else {
      throw SpeculativeDecodingError.invalidAcceptanceProbability(acceptanceProbability)
    }
    for (name, value) in [
      ("draftCostPerToken", draftCostPerToken),
      ("targetVerificationCost", targetVerificationCost),
      ("targetCostPerToken", targetCostPerToken),
    ] where !value.isFinite || value < 0 {
      throw SpeculativeDecodingError.invalidCost(name: name, value: value)
    }
    let expectedTokens = (0...maximumDraftTokens).reduce(0.0) {
      $0 + pow(acceptanceProbability, Double($1))
    }
    let speculativeCost = Double(maximumDraftTokens) * draftCostPerToken
      + targetVerificationCost
    let targetOnlyCost = expectedTokens * targetCostPerToken
    return SpeculativeCostEstimate(
      expectedEmittedTokens: expectedTokens,
      speculativeCost: speculativeCost,
      targetOnlyCostForExpectedTokens: targetOnlyCost,
      modeledSpeedup: speculativeCost == 0 ? .infinity : targetOnlyCost / speculativeCost)
  }
}

public enum P046SpeculativeDecodingContract {
  public static func validate(
    maximumDraftTokens: Int,
    vocabularySize: Int
  ) throws {
    guard maximumDraftTokens > 0 else {
      throw SpeculativeDecodingError.invalidDraftLength(maximumDraftTokens)
    }
    guard vocabularySize > 0 else {
      throw SpeculativeDecodingError.invalidVocabularySize(vocabularySize)
    }
  }
}

public enum P046SpeculativeDecodingJudge {
  public static func evaluate(
    _ implementation: SpeculativeDecodingImplementation
  ) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []

    do {
      let initialPrefix = [9]
      let draft: TokenDistributionProvider = { _ in .probabilities([1, 0]) }
      let target: TokenDistributionProvider = { prefix in
        prefix.count < initialPrefix.count + 2
          ? .probabilities([1, 0])
          : .probabilities([0, 1])
      }
      var generator = SeededGenerator(seed: 46)
      let result = try implementation(initialPrefix, 2, 2, draft, target, &generator)
      if result.emittedTokenIDs == [0, 0, 1],
        result.proposals.map(\.draftProbability) == [1, 1],
        result.verifications.map(\.acceptanceRatio) == [1, 1],
        result.verifications.allSatisfy(\.accepted),
        result.bonus?.tokenID == 1,
        result.rejectedAtProposalIndex == nil,
        result.draftEvaluationCount == 2,
        result.targetEvaluationCount == 3
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "accept all proposals and emit target bonus",
          message: "accept-all trace, evaluation counts, or bonus token is incorrect"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "accept all", message: error.localizedDescription))
    }

    do {
      let draft: TokenDistributionProvider = { _ in .probabilities([1, 0]) }
      let target: TokenDistributionProvider = { _ in .probabilities([0, 1]) }
      var generator = SeededGenerator(seed: 460)
      let result = try implementation([], 2, 2, draft, target, &generator)
      let rejection = result.verifications.first
      if result.emittedTokenIDs == [1],
        result.proposals.map(\.tokenID) == [0, 0],
        result.verifications.count == 1,
        rejection?.acceptanceRatio == 0,
        rejection?.accepted == false,
        rejection?.rejectionDistribution == [0, 1],
        rejection?.replacementTokenID == 1,
        result.rejectedAtProposalIndex == 0,
        result.bonus == nil,
        result.draftEvaluationCount == 2,
        result.targetEvaluationCount == 1
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "reject and sample correction distribution",
          message: "rejection must sample normalized max(p - q, 0) and stop the block"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "reject correction", message: error.localizedDescription))
    }

    do {
      let target = [0.25, 0.75]
      let draft = [0.75, 0.25]
      let output = try P046DistributionMath.oneStepOutputDistribution(
        target: target, draft: draft)
      if zip(output, target).allSatisfy({ abs($0 - $1) < 1e-12 }) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "enumerable one-step output distribution",
          message: "accept/reject/correction mass must reconstruct the target distribution"))
      }
    } catch {
      failures.append(JudgeFailure(
        caseName: "enumerable distribution", message: error.localizedDescription))
    }

    do {
      let estimate = try P046SpeculativeCostModel.estimate(
        maximumDraftTokens: 4,
        acceptanceProbability: 0.8,
        draftCostPerToken: 0.1,
        targetVerificationCost: 1,
        targetCostPerToken: 1)
      if estimate.expectedEmittedTokens > 3,
        estimate.modeledSpeedup > 1
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "conditional speedup model",
          message: "high acceptance and cheap draft work should satisfy the modeled break-even condition"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "cost model", message: error.localizedDescription))
    }

    passed += expectError(name: "reject K equals zero", failures: &failures) {
      var generator = SeededGenerator(seed: 1)
      _ = try implementation(
        [], 0, 2,
        { _ in .probabilities([0.5, 0.5]) },
        { _ in .probabilities([0.5, 0.5]) },
        &generator)
    }
    passed += expectError(name: "reject vocabulary mismatch", failures: &failures) {
      var generator = SeededGenerator(seed: 1)
      _ = try implementation(
        [], 1, 2,
        { _ in .probabilities([1, 0, 0]) },
        { _ in .probabilities([0.5, 0.5]) },
        &generator)
    }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 6, failures: failures)
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(caseName: name, message: "expected an error"))
      return 0
    } catch {
      return 1
    }
  }
}