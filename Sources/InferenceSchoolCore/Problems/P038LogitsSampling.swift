import Foundation

public enum LogitsSamplingError: Error, Equatable, LocalizedError {
  case emptyLogits
  case nonFiniteLogit(index: Int, value: Float)
  case invalidTemperature(Float)
  case invalidTopK(value: Int, vocabularySize: Int)
  case invalidTopP(Float)

  public var errorDescription: String? {
    switch self {
    case .emptyLogits:
      "Sampling requires at least one vocabulary logit."
    case .nonFiniteLogit(let index, let value):
      "Logit \(index) must be finite; received \(value)."
    case .invalidTemperature(let value):
      "Stochastic temperature must be finite and greater than zero; received \(value). Use greedy mode for temperature-zero behavior."
    case .invalidTopK(let value, let vocabularySize):
      "top-k must be in 1...\(vocabularySize); received \(value)."
    case .invalidTopP(let value):
      "top-p must be finite and in (0, 1]; received \(value)."
    }
  }
}

public struct SamplingConfiguration: Sendable, Equatable {
  public let temperature: Float
  public let topK: Int?
  public let topP: Float?

  public init(temperature: Float, topK: Int? = nil, topP: Float? = nil) {
    self.temperature = temperature
    self.topK = topK
    self.topP = topP
  }
}

public enum SamplingStrategy: Sendable, Equatable {
  case greedy
  case stochastic(SamplingConfiguration)
}

public struct SamplingCandidate: Sendable, Equatable {
  public let tokenID: Int
  public let logit: Float
  public let probability: Float

  public init(tokenID: Int, logit: Float, probability: Float) {
    self.tokenID = tokenID
    self.logit = logit
    self.probability = probability
  }
}

public struct SamplingTrace: Sendable, Equatable {
  public let selectedToken: Int
  public let retainedCandidates: [SamplingCandidate]
  public let randomDraw: Double?

  public init(
    selectedToken: Int,
    retainedCandidates: [SamplingCandidate],
    randomDraw: Double?
  ) {
    self.selectedToken = selectedToken
    self.retainedCandidates = retainedCandidates
    self.randomDraw = randomDraw
  }
}

public struct SeededGenerator: RandomNumberGenerator, Sendable, Equatable {
  public private(set) var state: UInt64

  public init(seed: UInt64) {
    state = seed
  }

  public mutating func next() -> UInt64 {
    state &+= 0x9e3779b97f4a7c15
    var value = state
    value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
    value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
    return value ^ (value >> 31)
  }

  public mutating func nextUnitInterval() -> Double {
    Double(next() >> 11) / 9_007_199_254_740_992.0
  }
}

public typealias LogitsSamplingImplementation = (
  _ logits: [Float],
  _ strategy: SamplingStrategy,
  _ generator: inout SeededGenerator
) throws -> SamplingTrace

public struct SeededLogitsSampler: Sendable, Equatable {
  public private(set) var generator: SeededGenerator

  public init(seed: UInt64) {
    generator = SeededGenerator(seed: seed)
  }

  public mutating func sample(
    logits: [Float],
    strategy: SamplingStrategy,
    using implementation: LogitsSamplingImplementation
  ) throws -> SamplingTrace {
    try implementation(logits, strategy, &generator)
  }
}

public enum P038LogitsSamplingContract {
  public static func validate(logits: [Float], strategy: SamplingStrategy) throws {
    guard !logits.isEmpty else { throw LogitsSamplingError.emptyLogits }
    for (index, value) in logits.enumerated() where !value.isFinite {
      throw LogitsSamplingError.nonFiniteLogit(index: index, value: value)
    }
    guard case .stochastic(let configuration) = strategy else { return }
    guard configuration.temperature.isFinite, configuration.temperature > 0 else {
      throw LogitsSamplingError.invalidTemperature(configuration.temperature)
    }
    if let topK = configuration.topK,
      !(1...logits.count).contains(topK)
    {
      throw LogitsSamplingError.invalidTopK(
        value: topK, vocabularySize: logits.count)
    }
    if let topP = configuration.topP,
      !topP.isFinite || topP <= 0 || topP > 1
    {
      throw LogitsSamplingError.invalidTopP(topP)
    }
  }
}

public enum P038LogitsSamplingJudge {
  public static let probabilityTolerance: Float = 2e-6

  public static func evaluate(_ implementation: LogitsSamplingImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      passed += compare(
        name: "greedy near-tie uses lowest token ID",
        logits: [1, 1, 0.999_999],
        strategy: .greedy,
        seed: 11,
        implementation: implementation,
        failures: &failures)
      passed += compare(
        name: "extreme logits use stable softmax",
        logits: [10_000, 0, -10_000],
        strategy: .stochastic(SamplingConfiguration(temperature: 0.5)),
        seed: 12,
        implementation: implementation,
        failures: &failures)
      passed += compare(
        name: "top-k then top-p composition",
        logits: [4, 3, 2, 1],
        strategy: .stochastic(SamplingConfiguration(temperature: 1, topK: 3, topP: 0.8)),
        seed: 13,
        implementation: implementation,
        failures: &failures)
      passed += compare(
        name: "top-p includes boundary-crossing token",
        logits: [0, 0, 0],
        strategy: .stochastic(SamplingConfiguration(temperature: 1, topP: 0.34)),
        seed: 14,
        implementation: implementation,
        failures: &failures)

      var first = SeededGenerator(seed: 0x1234)
      var second = SeededGenerator(seed: 0x1234)
      let strategy = SamplingStrategy.stochastic(
        SamplingConfiguration(temperature: 0.7, topK: 4, topP: 0.9))
      let firstTrace = try implementation([0.1, 0.2, 0.3, 0.4], strategy, &first)
      let secondTrace = try implementation([0.1, 0.2, 0.3, 0.4], strategy, &second)
      if tracesEqual(firstTrace, secondTrace), first == second {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "seeded reproducibility",
          message: "equal seeds must produce equal traces and next generator state"))
      }

      passed += expectError(name: "reject empty logits", failures: &failures) {
        var generator = SeededGenerator(seed: 1)
        _ = try implementation([], .greedy, &generator)
      }
      passed += expectError(name: "reject non-finite logits", failures: &failures) {
        var generator = SeededGenerator(seed: 1)
        _ = try implementation([0, .infinity], .greedy, &generator)
      }
      passed += expectError(name: "reject temperature zero", failures: &failures) {
        var generator = SeededGenerator(seed: 1)
        _ = try implementation(
          [0, 1],
          .stochastic(SamplingConfiguration(temperature: 0)),
          &generator)
      }
      passed += expectError(name: "reject top-k beyond vocabulary", failures: &failures) {
        var generator = SeededGenerator(seed: 1)
        _ = try implementation(
          [0, 1],
          .stochastic(SamplingConfiguration(temperature: 1, topK: 3)),
          &generator)
      }
      passed += expectError(name: "reject top-p outside range", failures: &failures) {
        var generator = SeededGenerator(seed: 1)
        _ = try implementation(
          [0, 1],
          .stochastic(SamplingConfiguration(temperature: 1, topP: 1.1)),
          &generator)
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 10, failures: failures)
  }

  private struct RankedLogit {
    let tokenID: Int
    let logit: Float
  }

  private static func compare(
    name: String,
    logits: [Float],
    strategy: SamplingStrategy,
    seed: UInt64,
    implementation: LogitsSamplingImplementation,
    failures: inout [JudgeFailure]
  ) -> Int {
    do {
      var actualGenerator = SeededGenerator(seed: seed)
      var expectedGenerator = SeededGenerator(seed: seed)
      let actual = try implementation(logits, strategy, &actualGenerator)
      let expected = try reference(logits, strategy, &expectedGenerator)
      guard tracesEqual(actual, expected), actualGenerator == expectedGenerator else {
        failures.append(JudgeFailure(
          caseName: name,
          message: "selected token, retained normalized probabilities, draw, or PRNG state differs"))
        return 0
      }
      return 1
    } catch {
      failures.append(JudgeFailure(caseName: name, message: error.localizedDescription))
      return 0
    }
  }

  private static func reference(
    _ logits: [Float],
    _ strategy: SamplingStrategy,
    _ generator: inout SeededGenerator
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
    if let topK = configuration.topK { ranked = Array(ranked.prefix(topK)) }
    let scaled = ranked.map { Double($0.logit) / Double(configuration.temperature) }
    let maximum = scaled.max()!
    let exponentials = scaled.map { exp($0 - maximum) }
    let denominator = exponentials.reduce(0, +)
    var probabilities = exponentials.map { $0 / denominator }
    if let topP = configuration.topP {
      var cumulative = 0.0
      var retained = 0
      repeat {
        cumulative += probabilities[retained]
        retained += 1
      } while retained < probabilities.count && cumulative < Double(topP)
      ranked = Array(ranked.prefix(retained))
      probabilities = Array(probabilities.prefix(retained))
    }
    let retainedSum = probabilities.reduce(0, +)
    probabilities = probabilities.map { $0 / retainedSum }
    let draw = generator.nextUnitInterval()
    var cumulative = 0.0
    var selected = ranked.last!.tokenID
    for index in ranked.indices {
      cumulative += probabilities[index]
      if draw < cumulative {
        selected = ranked[index].tokenID
        break
      }
    }
    return SamplingTrace(
      selectedToken: selected,
      retainedCandidates: ranked.indices.map {
        SamplingCandidate(
          tokenID: ranked[$0].tokenID,
          logit: ranked[$0].logit,
          probability: Float(probabilities[$0]))
      },
      randomDraw: draw)
  }

  private static func tracesEqual(_ lhs: SamplingTrace, _ rhs: SamplingTrace) -> Bool {
    lhs.selectedToken == rhs.selectedToken
      && lhs.randomDraw == rhs.randomDraw
      && lhs.retainedCandidates.count == rhs.retainedCandidates.count
      && zip(lhs.retainedCandidates, rhs.retainedCandidates).allSatisfy {
        $0.tokenID == $1.tokenID
          && $0.logit == $1.logit
          && abs($0.probability - $1.probability) <= probabilityTolerance
      }
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(
        caseName: name, message: "expected an error, but the sampler returned"))
      return 0
    } catch {
      return 1
    }
  }
}