import Foundation
import InferenceSchoolCore

public enum P038LogitsSamplingSolution {
  public static func sample(
    logits: [Float],
    strategy: SamplingStrategy,
    generator: inout SeededGenerator
  ) throws -> SamplingTrace {
    try P038LogitsSamplingContract.validate(logits: logits, strategy: strategy)
    var ranked = logits.indices.map { RankedLogit(tokenID: $0, logit: logits[$0]) }
    ranked.sort {
      $0.logit == $1.logit ? $0.tokenID < $1.tokenID : $0.logit > $1.logit
    }
    guard case .stochastic(let configuration) = strategy else {
      let winner = ranked[0]
      return SamplingTrace(
        selectedToken: winner.tokenID,
        retainedCandidates: [SamplingCandidate(
          tokenID: winner.tokenID, logit: winner.logit, probability: 1)],
        randomDraw: nil)
    }

    if let topK = configuration.topK {
      ranked = Array(ranked.prefix(topK))
    }
    let scaledLogits = ranked.map {
      Double($0.logit) / Double(configuration.temperature)
    }
    let maximum = scaledLogits.max()!
    let exponentials = scaledLogits.map { exp($0 - maximum) }
    let denominator = exponentials.reduce(0, +)
    var probabilities = exponentials.map { $0 / denominator }

    if let topP = configuration.topP {
      var cumulative = 0.0
      var retainedCount = 0
      repeat {
        cumulative += probabilities[retainedCount]
        retainedCount += 1
      } while retainedCount < probabilities.count && cumulative < Double(topP)
      ranked = Array(ranked.prefix(retainedCount))
      probabilities = Array(probabilities.prefix(retainedCount))
    }

    let retainedSum = probabilities.reduce(0, +)
    probabilities = probabilities.map { $0 / retainedSum }
    let draw = generator.nextUnitInterval()
    var cumulative = 0.0
    var selectedToken = ranked.last!.tokenID
    for index in ranked.indices {
      cumulative += probabilities[index]
      if draw < cumulative {
        selectedToken = ranked[index].tokenID
        break
      }
    }
    return SamplingTrace(
      selectedToken: selectedToken,
      retainedCandidates: ranked.indices.map {
        SamplingCandidate(
          tokenID: ranked[$0].tokenID,
          logit: ranked[$0].logit,
          probability: Float(probabilities[$0]))
      },
      randomDraw: draw)
  }

  private struct RankedLogit {
    let tokenID: Int
    let logit: Float
  }
}