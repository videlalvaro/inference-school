import Foundation

public enum P021SlidingWindowAttentionJudge {
  public static func evaluate(_ implementation: WindowedAttentionImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let c = try AttentionConfiguration(queryHeadCount: 2, keyValueHeadCount: 2, headDimension: 2)
      let q = try FloatTensor(attentionValues(count: 12, salt: 3), shape: [3, 2, 2])
      let k = try FloatTensor(attentionValues(count: 12, salt: 5), shape: [3, 2, 2])
      let v = try FloatTensor(attentionValues(count: 12, salt: 7), shape: [3, 2, 2])
      for (name, window) in [
        ("window one selects current token", 1), ("window at least context equals causal", 8),
      ] {
        let actual = try implementation(q, k, v, c, window)
        let expected = try AttentionJudgeOracle.materialized(
          queries: q, keys: k, values: v, configuration: c, window: window)
        if AttentionJudgeOracle.approximatelyEqual(actual, expected) {
          passed += 1
        } else {
          failures.append(
            JudgeFailure(
              caseName: name, message: "windowed output differs from inclusive-bound oracle"))
        }
      }
      let offsetC = try AttentionConfiguration(
        queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2, queryPositionOffset: 5,
        keyPositionOffset: 2)
      let oq = try FloatTensor([1, 0, 0, 1], shape: [2, 1, 2])
      let ok = try FloatTensor(attentionValues(count: 12, salt: 5), shape: [6, 1, 2])
      let ov = try FloatTensor(attentionValues(count: 12, salt: 7), shape: [6, 1, 2])
      let actual = try implementation(oq, ok, ov, offsetC, 3)
      let expected = try AttentionJudgeOracle.materialized(
        queries: oq, keys: ok, values: ov, configuration: offsetC, window: 3)
      if AttentionJudgeOracle.approximatelyEqual(actual, expected) {
        passed += 1
      } else {
        failures.append(
          JudgeFailure(
            caseName: "nonzero absolute position offsets",
            message: "window bounds used local rather than absolute positions"))
      }
      passed += AttentionJudgeOracle.expectError(name: "reject zero window", failures: &failures) {
        _ = try implementation(q, k, v, c, 0)
      }
    } catch {
      failures.append(
        JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }
}
