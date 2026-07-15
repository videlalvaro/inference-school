import Foundation

public enum P020TiledAttentionJudge {
  public static func evaluate(_ implementation: AttentionImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let configurations = [
        try AttentionConfiguration(queryHeadCount: 2, keyValueHeadCount: 2, headDimension: 3),
        try AttentionConfiguration(queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 4),
      ]
      let lengths = [5, 17]
      for index in configurations.indices {
        let c = configurations[index]
        let length = lengths[index]
        let q = try FloatTensor(
          attentionValues(count: length * c.queryHeadCount * c.headDimension, salt: 3),
          shape: [length, c.queryHeadCount, c.headDimension])
        let k = try FloatTensor(
          attentionValues(count: length * c.keyValueHeadCount * c.headDimension, salt: 5),
          shape: [length, c.keyValueHeadCount, c.headDimension])
        let v = try FloatTensor(
          attentionValues(count: length * c.keyValueHeadCount * c.headDimension, salt: 7),
          shape: [length, c.keyValueHeadCount, c.headDimension])
        let actual = try implementation(q, k, v, c)
        let expected = try AttentionJudgeOracle.materialized(
          queries: q, keys: k, values: v, configuration: c)
        if AttentionJudgeOracle.approximatelyEqual(
          actual, expected, absoluteTolerance: 6e-5, relativeTolerance: 1.2e-4)
        {
          passed += 1
        } else {
          failures.append(
            JudgeFailure(
              caseName: index == 0 ? "partial five-key tile" : "crosses sixteen-key tile",
              message: "tiled output differs from materialized oracle"))
        }
      }
      let c = configurations[0]
      let valid = try FloatTensor(Array(repeating: 1, count: 6), shape: [1, 2, 3])
      passed += AttentionJudgeOracle.expectError(
        name: "reject key/value length mismatch", failures: &failures
      ) {
        _ = try implementation(
          valid, valid, FloatTensor(Array(repeating: 1, count: 12), shape: [2, 2, 3]), c)
      }
    } catch {
      failures.append(
        JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 3, failures: failures)
  }
}
