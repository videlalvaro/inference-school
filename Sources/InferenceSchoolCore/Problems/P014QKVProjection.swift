import Foundation

public enum QKVProjectionError: Error, Equatable, LocalizedError {
  case hiddenWidthMismatch(expected: Int, actual: Int)
  case nonFiniteValue(tensor: String, linearIndex: Int)

  public var errorDescription: String? {
    switch self {
    case .hiddenWidthMismatch(let expected, let actual):
      "Hidden width must be \(expected); received \(actual)."
    case .nonFiniteValue(let tensor, let linearIndex):
      "\(tensor) contains a non-finite value at linear index \(linearIndex)."
    }
  }
}

public struct QKVProjectionResult: Sendable, Equatable {
  public let queries: FloatTensor
  public let keys: FloatTensor
  public let values: FloatTensor

  public init(queries: FloatTensor, keys: FloatTensor, values: FloatTensor) {
    self.queries = queries
    self.keys = keys
    self.values = values
  }
}

public typealias QKVProjectionImplementation = (
  _ hidden: FloatTensor,
  _ queryWeights: FloatTensor,
  _ keyWeights: FloatTensor,
  _ valueWeights: FloatTensor,
  _ configuration: AttentionConfiguration
) throws -> QKVProjectionResult

public enum QKVProjectionContract {
  public static func validate(
    hidden: FloatTensor,
    queryWeights: FloatTensor,
    keyWeights: FloatTensor,
    valueWeights: FloatTensor,
    configuration: AttentionConfiguration
  ) throws {
    guard hidden.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: hidden.rank)
    }
    let modelDimension = hidden.shape[1]
    let expectedQueryShape = [
      modelDimension,
      configuration.queryHeadCount * configuration.headDimension,
    ]
    let expectedKeyValueShape = [
      modelDimension,
      configuration.keyValueHeadCount * configuration.headDimension,
    ]
    for (name, tensor, expected) in [
      ("Query weights", queryWeights, expectedQueryShape),
      ("Key weights", keyWeights, expectedKeyValueShape),
      ("Value weights", valueWeights, expectedKeyValueShape),
    ] where tensor.shape != expected {
      throw AttentionError.shapeMismatch(tensor: name, expected: expected, actual: tensor.shape)
    }
    for (name, tensor) in [
      ("Hidden states", hidden),
      ("Query weights", queryWeights),
      ("Key weights", keyWeights),
      ("Value weights", valueWeights),
    ] {
      if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
        throw QKVProjectionError.nonFiniteValue(tensor: name, linearIndex: index)
      }
    }
  }
}

public enum P014QKVProjectionJudge {
  public static let absoluteTolerance: Float = 2e-5
  public static let relativeTolerance: Float = 5e-5

  public static func evaluate(_ implementation: QKVProjectionImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let configurations = [
        try AttentionConfiguration(queryHeadCount: 2, keyValueHeadCount: 1, headDimension: 2),
        try AttentionConfiguration(queryHeadCount: 4, keyValueHeadCount: 2, headDimension: 1),
      ]
      let hiddenCases = [
        try FloatTensor([1, 2, -1, 0.5, 3, 2], shape: [2, 3]),
        try FloatTensor([], shape: [0, 2]),
      ]
      for index in configurations.indices {
        let configuration = configurations[index]
        let hidden = hiddenCases[index]
        let modelDimension = hidden.shape[1]
        let queryColumns = configuration.queryHeadCount * configuration.headDimension
        let keyValueColumns = configuration.keyValueHeadCount * configuration.headDimension
        let queryWeights = try FloatTensor(
          deterministicValues(count: modelDimension * queryColumns, salt: 3),
          shape: [modelDimension, queryColumns]
        )
        let keyWeights = try FloatTensor(
          deterministicValues(count: modelDimension * keyValueColumns, salt: 5),
          shape: [modelDimension, keyValueColumns]
        )
        let valueWeights = try FloatTensor(
          deterministicValues(count: modelDimension * keyValueColumns, salt: 7),
          shape: [modelDimension, keyValueColumns]
        )
        let actual = try implementation(
          hidden,
          queryWeights,
          keyWeights,
          valueWeights,
          configuration
        )
        let expected = try reference(
          hidden: hidden,
          queryWeights: queryWeights,
          keyWeights: keyWeights,
          valueWeights: valueWeights,
          configuration: configuration
        )
        if equal(actual.queries, expected.queries),
          equal(actual.keys, expected.keys),
          equal(actual.values, expected.values)
        {
          passed += 1
        } else {
          failures.append(
            JudgeFailure(
              caseName: index == 0 ? "separate query and KV head counts" : "empty sequence",
              message: "projection values or [S,H,dh] shapes do not match the independent reference"
            ))
        }
      }

      let configuration = configurations[0]
      let hidden = try FloatTensor([1, 2, 3], shape: [1, 3])
      let validKV = try FloatTensor([1, 2, 3, 4, 5, 6], shape: [3, 2])
      passed += AttentionJudgeOracle.expectError(
        name: "reject wrong query projection width", failures: &failures
      ) {
        _ = try implementation(
          hidden,
          FloatTensor([1, 2, 3], shape: [3, 1]),
          validKV,
          validKV,
          configuration
        )
      }
      passed += AttentionJudgeOracle.expectError(
        name: "reject non-finite hidden state", failures: &failures
      ) {
        _ = try implementation(
          FloatTensor([.infinity, 0, 1], shape: [1, 3]),
          FloatTensor(Array(repeating: 1, count: 12), shape: [3, 4]),
          validKV,
          validKV,
          configuration
        )
      }
    } catch {
      failures.append(
        JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }

  private static func deterministicValues(count: Int, salt: Int) -> [Float] {
    (0..<count).map { Float((($0 * salt) % 11) - 5) / 4 }
  }

  private static func reference(
    hidden: FloatTensor,
    queryWeights: FloatTensor,
    keyWeights: FloatTensor,
    valueWeights: FloatTensor,
    configuration: AttentionConfiguration
  ) throws -> QKVProjectionResult {
    func project(_ weights: FloatTensor, heads: Int) throws -> FloatTensor {
      let sequenceLength = hidden.shape[0]
      let modelDimension = hidden.shape[1]
      let columns = heads * configuration.headDimension
      var output = Array(repeating: Float.zero, count: sequenceLength * columns)
      for sequence in 0..<sequenceLength {
        for column in 0..<columns {
          var sum = 0.0
          for model in 0..<modelDimension {
            sum +=
              Double(hidden.storage[sequence * modelDimension + model])
              * Double(weights.storage[model * columns + column])
          }
          output[sequence * columns + column] = Float(sum)
        }
      }
      return try FloatTensor(
        output,
        shape: [sequenceLength, heads, configuration.headDimension]
      )
    }
    return QKVProjectionResult(
      queries: try project(queryWeights, heads: configuration.queryHeadCount),
      keys: try project(keyWeights, heads: configuration.keyValueHeadCount),
      values: try project(valueWeights, heads: configuration.keyValueHeadCount)
    )
  }

  private static func equal(_ lhs: FloatTensor, _ rhs: FloatTensor) -> Bool {
    lhs.shape == rhs.shape
      && zip(lhs.storage, rhs.storage).allSatisfy { actual, expected in
        abs(actual - expected) <= absoluteTolerance + relativeTolerance * abs(expected)
      }
  }
}
