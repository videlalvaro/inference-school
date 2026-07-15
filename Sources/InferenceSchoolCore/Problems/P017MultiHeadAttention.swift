import Foundation

public enum MultiHeadAttentionError: Error, Equatable, LocalizedError {
  case requiresEqualHeadCounts(queryHeads: Int, keyValueHeads: Int)

  public var errorDescription: String? {
    switch self {
    case .requiresEqualHeadCounts(let queryHeads, let keyValueHeads):
      "Problem 017 requires equal query and key/value head counts; received \(queryHeads) and \(keyValueHeads)."
    }
  }
}

public enum P017MultiHeadAttentionContract {
  public static func validate(
    _ queries: FloatTensor, _ keys: FloatTensor, _ values: FloatTensor,
    _ configuration: AttentionConfiguration
  ) throws -> AttentionInput {
    guard configuration.queryHeadCount == configuration.keyValueHeadCount else {
      throw MultiHeadAttentionError.requiresEqualHeadCounts(
        queryHeads: configuration.queryHeadCount, keyValueHeads: configuration.keyValueHeadCount)
    }
    let input = try AttentionInput(
      queries: queries, keys: keys, values: values, configuration: configuration)
    try validateVisibleKeys(input)
    return input
  }
}

public func validateVisibleKeys(_ input: AttentionInput, window: Int? = nil) throws {
  if let window, window <= 0 { throw AttentionError.invalidWindow(window) }
  for query in 0..<input.queryLength {
    let queryPosition = input.configuration.queryPositionOffset + query
    let visible = (0..<input.keyValueLength).contains { key in
      let keyPosition = input.configuration.keyPositionOffset + key
      guard keyPosition <= queryPosition else { return false }
      guard let window else { return true }
      return keyPosition >= queryPosition - window + 1
    }
    guard visible else { throw AttentionError.noVisibleKeys(queryPosition: queryPosition) }
  }
}

public enum P017MultiHeadAttentionJudge {
  public static func evaluate(_ implementation: AttentionImplementation) -> JudgeReport {
    do {
      return evaluateAttentionImplementation(
        implementation,
        configurations: [
          try AttentionConfiguration(queryHeadCount: 3, keyValueHeadCount: 3, headDimension: 2),
          try AttentionConfiguration(
            queryHeadCount: 2, keyValueHeadCount: 2, headDimension: 3, queryPositionOffset: 2),
        ], caseNames: ["three independent heads", "multi-head decode offsets"])
    } catch {
      return JudgeReport(
        passedCaseCount: 0,
        totalCaseCount: 3,
        failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)]
      )
    }
  }
}

func evaluateAttentionImplementation(
  _ implementation: AttentionImplementation, configurations: [AttentionConfiguration],
  caseNames: [String]
) -> JudgeReport {
  var failures: [JudgeFailure] = []
  var passed = 0
  do {
    for (caseIndex, configuration) in configurations.enumerated() {
      let queryLength = caseIndex + 3
      let keyLength = configuration.queryPositionOffset == 0 ? queryLength : queryLength + 2
      let queries = try FloatTensor(
        attentionValues(
          count: queryLength * configuration.queryHeadCount * configuration.headDimension, salt: 3),
        shape: [queryLength, configuration.queryHeadCount, configuration.headDimension])
      let keys = try FloatTensor(
        attentionValues(
          count: keyLength * configuration.keyValueHeadCount * configuration.headDimension, salt: 5),
        shape: [keyLength, configuration.keyValueHeadCount, configuration.headDimension])
      let values = try FloatTensor(
        attentionValues(
          count: keyLength * configuration.keyValueHeadCount * configuration.headDimension, salt: 7),
        shape: [keyLength, configuration.keyValueHeadCount, configuration.headDimension])
      let actual = try implementation(queries, keys, values, configuration)
      let expected = try AttentionJudgeOracle.materialized(
        queries: queries, keys: keys, values: values, configuration: configuration)
      if AttentionJudgeOracle.approximatelyEqual(actual, expected) {
        passed += 1
      } else {
        failures.append(
          JudgeFailure(
            caseName: caseNames[caseIndex],
            message: "per-head attention differs from the materialized Double oracle"))
      }
    }
    let configuration = configurations[0]
    let valid = try FloatTensor(
      Array(repeating: 1, count: configuration.queryHeadCount * configuration.headDimension),
      shape: [1, configuration.queryHeadCount, configuration.headDimension])
    passed += AttentionJudgeOracle.expectError(
      name: "reject KV shape inconsistent with configuration", failures: &failures
    ) {
      _ = try implementation(
        valid, FloatTensor([1, 2], shape: [1, 1, 2]), FloatTensor([1, 2], shape: [1, 1, 2]),
        configuration)
    }
  } catch {
    failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
  }
  return JudgeReport(
    passedCaseCount: passed, totalCaseCount: configurations.count + 1, failures: failures)
}

func attentionValues(count: Int, salt: Int) -> [Float] {
  (0..<count).map { Float((($0 * salt) % 17) - 8) / 5 }
}
