import Foundation

public enum P019OnlineAttentionJudge {
  public static func evaluate(_ implementation: AttentionImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let cases = [
        try makeCase(
          name: "online recurrence across five keys", sequenceLength: 5, headDimension: 3, scale: 1),
        try makeCase(
          name: "large logits across thirty-three keys", sequenceLength: 33, headDimension: 4,
          scale: 80),
        try makeDecodeCase(),
      ]
      for testCase in cases {
        let actual = try implementation(
          testCase.queries, testCase.keys, testCase.values, testCase.configuration)
        let expected = try AttentionJudgeOracle.materialized(
          queries: testCase.queries, keys: testCase.keys, values: testCase.values,
          configuration: testCase.configuration)
        if AttentionJudgeOracle.approximatelyEqual(
          actual, expected, absoluteTolerance: 5e-5, relativeTolerance: 1e-4)
        {
          passed += 1
        } else {
          failures.append(
            JudgeFailure(
              caseName: testCase.name,
              message: "online output differs from the Problem 016 materialized oracle"))
        }
      }
      let configuration = try AttentionConfiguration(
        queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2)
      let valid = try FloatTensor([1, 2], shape: [1, 1, 2])
      passed += AttentionJudgeOracle.expectError(
        name: "reject mismatched value shape", failures: &failures
      ) { _ = try implementation(valid, valid, FloatTensor([1], shape: [1, 1, 1]), configuration) }
    } catch {
      failures.append(
        JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }

  private struct ValueCase {
    let name: String
    let queries: FloatTensor
    let keys: FloatTensor
    let values: FloatTensor
    let configuration: AttentionConfiguration
  }
  private static func makeCase(name: String, sequenceLength: Int, headDimension: Int, scale: Float)
    throws -> ValueCase
  {
    ValueCase(
      name: name,
      queries: try FloatTensor(
        attentionValues(count: sequenceLength * headDimension, salt: 3).map { $0 * scale },
        shape: [sequenceLength, 1, headDimension]),
      keys: try FloatTensor(
        attentionValues(count: sequenceLength * headDimension, salt: 5).map { $0 * scale },
        shape: [sequenceLength, 1, headDimension]),
      values: try FloatTensor(
        attentionValues(count: sequenceLength * headDimension, salt: 7),
        shape: [sequenceLength, 1, headDimension]),
      configuration: try AttentionConfiguration(
        queryHeadCount: 1, keyValueHeadCount: 1, headDimension: headDimension)
    )
  }
  private static func makeDecodeCase() throws -> ValueCase {
    ValueCase(
      name: "online decode offsets",
      queries: try FloatTensor([1, 0, 0, 1], shape: [2, 1, 2]),
      keys: try FloatTensor([1, 0, 0, 1, 1, 1, -1, 1], shape: [4, 1, 2]),
      values: try FloatTensor([1, 2, 3, 4, 5, 6, 7, 8], shape: [4, 1, 2]),
      configuration: try AttentionConfiguration(
        queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2, queryPositionOffset: 2)
    )
  }
}
