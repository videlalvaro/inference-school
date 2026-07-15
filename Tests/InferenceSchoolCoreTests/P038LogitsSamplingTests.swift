import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P038LogitsSamplingTests: XCTestCase {
  func testCanonicalSamplerPassesJudge() {
    let report = P038LogitsSamplingJudge.evaluate(P038LogitsSamplingSolution.sample)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsAlwaysGreedySampler() {
    let report = P038LogitsSamplingJudge.evaluate { logits, strategy, generator in
      try P038LogitsSamplingContract.validate(logits: logits, strategy: strategy)
      _ = generator
      let winner = logits.indices.min {
        logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
      }!
      return SamplingTrace(
        selectedToken: winner,
        retainedCandidates: [SamplingCandidate(
          tokenID: winner, logit: logits[winner], probability: 1)],
        randomDraw: nil)
    }
    XCTAssertFalse(report.isPassing)
  }

  func testTopPIncludesCrossingTokenAndRenormalizes() throws {
    var generator = SeededGenerator(seed: 7)
    let trace = try P038LogitsSamplingSolution.sample(
      logits: [0, 0, 0],
      strategy: .stochastic(SamplingConfiguration(
        temperature: 1, topP: 0.34)),
      generator: &generator)
    XCTAssertEqual(trace.retainedCandidates.map(\.tokenID), [0, 1])
    XCTAssertEqual(trace.retainedCandidates.map(\.probability), [0.5, 0.5])
    XCTAssertNotNil(trace.randomDraw)
  }

  func testSeededSamplerOwnsReproducibleState() throws {
    var first = SeededLogitsSampler(seed: 42)
    var second = SeededLogitsSampler(seed: 42)
    let strategy = SamplingStrategy.stochastic(
      SamplingConfiguration(temperature: 0.8, topK: 3, topP: 0.9))
    let firstTrace = try first.sample(
      logits: [0.1, 0.2, 0.3, 0.4],
      strategy: strategy,
      using: P038LogitsSamplingSolution.sample)
    let secondTrace = try second.sample(
      logits: [0.1, 0.2, 0.3, 0.4],
      strategy: strategy,
      using: P038LogitsSamplingSolution.sample)
    XCTAssertEqual(firstTrace, secondTrace)
    XCTAssertEqual(first, second)
  }

  func testGreedyIsTemperatureZeroPolicyAndUsesStableTieOrder() throws {
    var generator = SeededGenerator(seed: 1)
    let greedy = try P038LogitsSamplingSolution.sample(
      logits: [2, 2, -1000], strategy: .greedy, generator: &generator)
    XCTAssertEqual(greedy.selectedToken, 0)
    XCTAssertNil(greedy.randomDraw)
    XCTAssertThrowsError(try P038LogitsSamplingSolution.sample(
      logits: [2, 2],
      strategy: .stochastic(SamplingConfiguration(temperature: 0)),
      generator: &generator))
  }
}