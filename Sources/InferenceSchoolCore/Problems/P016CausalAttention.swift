import Foundation

public enum CausalAttentionError: Error, Equatable, LocalizedError {
  case requiresSingleHead(queryHeads: Int, keyValueHeads: Int)

  public var errorDescription: String? {
    switch self {
    case .requiresSingleHead(let queryHeads, let keyValueHeads):
      "Problem 016 requires one query head and one key/value head; received \(queryHeads) and \(keyValueHeads)."
    }
  }
}

public enum P016CausalAttentionContract {
  public static func validate(
    queries: FloatTensor,
    keys: FloatTensor,
    values: FloatTensor,
    configuration: AttentionConfiguration
  ) throws -> AttentionInput {
    guard configuration.queryHeadCount == 1, configuration.keyValueHeadCount == 1 else {
      throw CausalAttentionError.requiresSingleHead(
        queryHeads: configuration.queryHeadCount,
        keyValueHeads: configuration.keyValueHeadCount
      )
    }
    let input = try AttentionInput(
      queries: queries, keys: keys, values: values, configuration: configuration)
    for query in 0..<input.queryLength {
      let queryPosition = configuration.queryPositionOffset + query
      let hasVisibleKey = (0..<input.keyValueLength).contains { key in
        configuration.keyPositionOffset + key <= queryPosition
      }
      guard hasVisibleKey else { throw AttentionError.noVisibleKeys(queryPosition: queryPosition) }
    }
    return input
  }
}

public enum P016CausalAttentionJudge {
  public static func evaluate(_ implementation: AttentionImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let cases = try makeCases()
      for testCase in cases {
        let actual = try implementation(
          testCase.queries, testCase.keys, testCase.values, testCase.configuration)
        let expected = try AttentionJudgeOracle.materialized(
          queries: testCase.queries,
          keys: testCase.keys,
          values: testCase.values,
          configuration: testCase.configuration
        )
        if AttentionJudgeOracle.approximatelyEqual(actual, expected) {
          passed += 1
        } else {
          failures.append(
            JudgeFailure(
              caseName: testCase.name,
              message:
                "expected \(expected.storage), received shape \(actual.shape) and values \(actual.storage)"
            ))
        }
      }
      let configuration = try AttentionConfiguration(
        queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2)
      let valid = try FloatTensor([1, 2], shape: [1, 1, 2])
      passed += AttentionJudgeOracle.expectError(
        name: "reject wrong query shape", failures: &failures
      ) {
        _ = try implementation(FloatTensor([1, 2], shape: [1, 2]), valid, valid, configuration)
      }
      let noVisibleConfiguration = try AttentionConfiguration(
        queryHeadCount: 1,
        keyValueHeadCount: 1,
        headDimension: 2,
        queryPositionOffset: 0,
        keyPositionOffset: 1
      )
      passed += AttentionJudgeOracle.expectError(
        name: "reject query with no visible keys", failures: &failures
      ) {
        _ = try implementation(valid, valid, valid, noVisibleConfiguration)
      }
    } catch {
      failures.append(
        JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 5, failures: failures)
  }

  private struct ValueCase {
    let name: String
    let queries: FloatTensor
    let keys: FloatTensor
    let values: FloatTensor
    let configuration: AttentionConfiguration
  }

  private static func makeCases() throws -> [ValueCase] {
    [
      ValueCase(
        name: "three causal rows",
        queries: try FloatTensor([1, 0, 0, 1, 1, 1], shape: [3, 1, 2]),
        keys: try FloatTensor([1, 0, 0, 1, -1, 1], shape: [3, 1, 2]),
        values: try FloatTensor([2, 1, 4, -1, 8, 3], shape: [3, 1, 2]),
        configuration: try AttentionConfiguration(
          queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2)
      ),
      ValueCase(
        name: "stable difficult logits",
        queries: try FloatTensor([1_000, 0, 1_001, 0], shape: [2, 1, 2]),
        keys: try FloatTensor([1_000, 0, 999, 0], shape: [2, 1, 2]),
        values: try FloatTensor([1, 2, -3, 4], shape: [2, 1, 2]),
        configuration: try AttentionConfiguration(
          queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2)
      ),
      ValueCase(
        name: "decode offsets and longer KV context",
        queries: try FloatTensor([1, -1, 0.5, 2], shape: [2, 1, 2]),
        keys: try FloatTensor([1, 0, 0, 1, 1, 1, -1, 0.5], shape: [4, 1, 2]),
        values: try FloatTensor([1, 2, 3, 4, 5, 6, 7, 8], shape: [4, 1, 2]),
        configuration: try AttentionConfiguration(
          queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 2, queryPositionOffset: 2)
      ),
    ]
  }
}
