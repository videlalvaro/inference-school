import InferenceSchoolCore

public enum P038LogitsSamplingExercise {
  public static func sample(
    logits: [Float],
    strategy: SamplingStrategy,
    generator: inout SeededGenerator
  ) throws -> SamplingTrace {
    try P038LogitsSamplingContract.validate(logits: logits, strategy: strategy)
    _ = generator
    let winner = logits.indices.min {
      logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
    }!
    // TODO: implement temperature scaling, top-k, top-p, renormalization, and seeded draw.
    return SamplingTrace(
      selectedToken: winner,
      retainedCandidates: [SamplingCandidate(
        tokenID: winner, logit: logits[winner], probability: 1)],
      randomDraw: nil)
  }
}