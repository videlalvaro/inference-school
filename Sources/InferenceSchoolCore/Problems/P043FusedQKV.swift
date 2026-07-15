import Foundation

public enum FusedQKVError: Error, Equatable, LocalizedError {
  case emptySequence
  case sequenceLengthExceedsMaximum(maximum: Int, actual: Int)
  case modelDimensionExceedsMaximum(maximum: Int, actual: Int)
  case projectionDimensionExceedsMaximum(maximum: Int, actual: Int)
  case shapeMismatch(tensor: String, expected: [Int], actual: [Int])
  case invalidEpsilon(Float)
  case nonFiniteValue(tensor: String, linearIndex: Int)

  public var errorDescription: String? {
    switch self {
    case .emptySequence:
      "Fused RMSNorm plus Q/K/V projection requires at least one token."
    case let .sequenceLengthExceedsMaximum(maximum, actual):
      "Sequence length \(actual) exceeds the educational fused-kernel maximum \(maximum)."
    case let .modelDimensionExceedsMaximum(maximum, actual):
      "Model dimension \(actual) exceeds the educational fused-kernel maximum \(maximum)."
    case let .projectionDimensionExceedsMaximum(maximum, actual):
      "Combined Q/K/V projection width \(actual) exceeds the educational fused-kernel maximum \(maximum)."
    case let .shapeMismatch(tensor, expected, actual):
      "\(tensor) must have shape \(expected); received \(actual)."
    case let .invalidEpsilon(value):
      "Epsilon must be finite and positive; received \(value)."
    case let .nonFiniteValue(tensor, linearIndex):
      "\(tensor) contains a non-finite value at linear index \(linearIndex)."
    }
  }
}

public struct FusedQKVRequest: Sendable, Equatable {
  public let input: FloatTensor
  public let gamma: FloatTensor
  public let queryWeights: FloatTensor
  public let keyWeights: FloatTensor
  public let valueWeights: FloatTensor
  public let epsilon: Float
  public let configuration: DecoderConfiguration

  public init(
    input: FloatTensor,
    gamma: FloatTensor,
    queryWeights: FloatTensor,
    keyWeights: FloatTensor,
    valueWeights: FloatTensor,
    epsilon: Float,
    configuration: DecoderConfiguration
  ) {
    self.input = input
    self.gamma = gamma
    self.queryWeights = queryWeights
    self.keyWeights = keyWeights
    self.valueWeights = valueWeights
    self.epsilon = epsilon
    self.configuration = configuration
  }
}

public struct FusedQKVResult: Sendable, Equatable {
  public let queries: FloatTensor
  public let keys: FloatTensor
  public let values: FloatTensor

  public init(queries: FloatTensor, keys: FloatTensor, values: FloatTensor) {
    self.queries = queries
    self.keys = keys
    self.values = values
  }
}

public struct FusedQKVCost: Sendable, Equatable {
  public let dispatchCount: Int
  public let logicalTensorBytes: Int
  public let intermediateBytes: Int

  public init(dispatchCount: Int, logicalTensorBytes: Int, intermediateBytes: Int) {
    self.dispatchCount = dispatchCount
    self.logicalTensorBytes = logicalTensorBytes
    self.intermediateBytes = intermediateBytes
  }
}

public struct FusedQKVCostComparison: Sendable, Equatable {
  public let separate: FusedQKVCost
  public let fused: FusedQKVCost

  public init(separate: FusedQKVCost, fused: FusedQKVCost) {
    self.separate = separate
    self.fused = fused
  }
}

public typealias FusedQKVImplementation = (FusedQKVRequest) throws -> FusedQKVResult

public enum P043FusedQKVContract {
  public static let maximumSequenceLength = 64
  public static let maximumModelDimension = 256
  public static let maximumCombinedProjectionDimension = 768

  public static func validate(_ request: FusedQKVRequest) throws {
    guard request.input.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: request.input.rank)
    }
    guard request.gamma.rank == 1 else {
      throw TensorError.rankMismatch(expected: 1, actual: request.gamma.rank)
    }
    guard request.queryWeights.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: request.queryWeights.rank)
    }
    guard request.keyWeights.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: request.keyWeights.rank)
    }
    guard request.valueWeights.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: request.valueWeights.rank)
    }

    let sequenceLength = request.input.shape[0]
    let modelDimension = request.configuration.modelDimension
    let queryDimension = request.configuration.queryProjectionDimension
    let keyValueDimension = request.configuration.keyValueProjectionDimension
    let combinedProjectionDimension = queryDimension + 2 * keyValueDimension

    guard sequenceLength > 0 else { throw FusedQKVError.emptySequence }
    guard sequenceLength <= maximumSequenceLength else {
      throw FusedQKVError.sequenceLengthExceedsMaximum(
        maximum: maximumSequenceLength, actual: sequenceLength)
    }
    guard modelDimension <= maximumModelDimension else {
      throw FusedQKVError.modelDimensionExceedsMaximum(
        maximum: maximumModelDimension, actual: modelDimension)
    }
    guard combinedProjectionDimension <= maximumCombinedProjectionDimension else {
      throw FusedQKVError.projectionDimensionExceedsMaximum(
        maximum: maximumCombinedProjectionDimension, actual: combinedProjectionDimension)
    }

    let shapes: [(String, FloatTensor, [Int])] = [
      ("Input", request.input, [sequenceLength, modelDimension]),
      ("Gamma", request.gamma, [modelDimension]),
      ("Query weights", request.queryWeights, [queryDimension, modelDimension]),
      ("Key weights", request.keyWeights, [keyValueDimension, modelDimension]),
      ("Value weights", request.valueWeights, [keyValueDimension, modelDimension]),
    ]
    for (name, tensor, expected) in shapes {
      guard tensor.shape == expected else {
        throw FusedQKVError.shapeMismatch(tensor: name, expected: expected, actual: tensor.shape)
      }
      if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
        throw FusedQKVError.nonFiniteValue(tensor: name, linearIndex: index)
      }
    }
    guard request.epsilon.isFinite, request.epsilon > 0 else {
      throw FusedQKVError.invalidEpsilon(request.epsilon)
    }
  }

  public static func validate(_ result: FusedQKVResult, for request: FusedQKVRequest) throws {
    let sequenceLength = request.input.shape[0]
    let headDimension = request.configuration.headDimension
    let expected: [(String, FloatTensor, [Int])] = [
      ("Queries", result.queries, [sequenceLength, request.configuration.queryHeadCount, headDimension]),
      ("Keys", result.keys, [sequenceLength, request.configuration.keyValueHeadCount, headDimension]),
      ("Values", result.values, [sequenceLength, request.configuration.keyValueHeadCount, headDimension]),
    ]
    for (name, tensor, shape) in expected {
      guard tensor.shape == shape else {
        throw FusedQKVError.shapeMismatch(tensor: name, expected: shape, actual: tensor.shape)
      }
      if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
        throw FusedQKVError.nonFiniteValue(tensor: name, linearIndex: index)
      }
    }
  }
}

public enum P043FusedQKVCostModel {
  public static func compare(_ request: FusedQKVRequest) throws -> FusedQKVCostComparison {
    try P043FusedQKVContract.validate(request)
    let floatBytes = MemoryLayout<Float>.stride
    let sequenceLength = request.input.shape[0]
    let modelDimension = request.configuration.modelDimension
    let queryDimension = request.configuration.queryProjectionDimension
    let keyValueDimension = request.configuration.keyValueProjectionDimension
    let normalizedBytes = sequenceLength * modelDimension * floatBytes
    let inputBytes = normalizedBytes
    let gammaBytes = modelDimension * floatBytes
    let weightBytes =
      (queryDimension + 2 * keyValueDimension) * modelDimension * floatBytes
    let outputBytes = sequenceLength * (queryDimension + 2 * keyValueDimension) * floatBytes
    return FusedQKVCostComparison(
      separate: FusedQKVCost(
        dispatchCount: 4,
        logicalTensorBytes: inputBytes + gammaBytes + 4 * normalizedBytes + weightBytes + outputBytes,
        intermediateBytes: normalizedBytes),
      fused: FusedQKVCost(
        dispatchCount: 1,
        logicalTensorBytes: inputBytes + gammaBytes + weightBytes + outputBytes,
        intermediateBytes: 0))
  }
}

public enum P043FusedQKVJudge {
  public static let absoluteTolerance: Float = 5e-5
  public static let relativeTolerance: Float = 1e-4

  public static func evaluate(_ implementation: FusedQKVImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      for (name, request) in try valueCases() {
        do {
          let actual = try implementation(request)
          try P043FusedQKVContract.validate(actual, for: request)
          let expected = try oracle(request)
          if equal(actual, expected) {
            passed += 1
          } else {
            failures.append(JudgeFailure(
              caseName: name,
              message: "fused Q/K/V values differ from the independent RMSNorm plus projection oracle"))
          }
        } catch {
          failures.append(JudgeFailure(caseName: name, message: error.localizedDescription))
        }
      }

      let base = try makeRequest(sequenceLength: 2, configuration: smallConfiguration())
      passed += expectError(name: "reject input rank", failures: &failures) {
        _ = try implementation(FusedQKVRequest(
          input: FloatTensor(base.input.storage, shape: [base.input.elementCount]),
          gamma: base.gamma,
          queryWeights: base.queryWeights,
          keyWeights: base.keyWeights,
          valueWeights: base.valueWeights,
          epsilon: base.epsilon,
          configuration: base.configuration))
      }
      passed += expectError(name: "reject empty sequence", failures: &failures) {
        _ = try implementation(FusedQKVRequest(
          input: FloatTensor([], shape: [0, base.configuration.modelDimension]),
          gamma: base.gamma,
          queryWeights: base.queryWeights,
          keyWeights: base.keyWeights,
          valueWeights: base.valueWeights,
          epsilon: base.epsilon,
          configuration: base.configuration))
      }
      passed += expectError(name: "reject gamma shape", failures: &failures) {
        _ = try implementation(FusedQKVRequest(
          input: base.input,
          gamma: FloatTensor([1, 1, 1], shape: [3]),
          queryWeights: base.queryWeights,
          keyWeights: base.keyWeights,
          valueWeights: base.valueWeights,
          epsilon: base.epsilon,
          configuration: base.configuration))
      }
      passed += expectError(name: "reject query weight shape", failures: &failures) {
        _ = try implementation(FusedQKVRequest(
          input: base.input,
          gamma: base.gamma,
          queryWeights: FloatTensor(Array(repeating: 0, count: 12), shape: [3, 4]),
          keyWeights: base.keyWeights,
          valueWeights: base.valueWeights,
          epsilon: base.epsilon,
          configuration: base.configuration))
      }
      passed += expectError(name: "reject epsilon", failures: &failures) {
        _ = try implementation(FusedQKVRequest(
          input: base.input,
          gamma: base.gamma,
          queryWeights: base.queryWeights,
          keyWeights: base.keyWeights,
          valueWeights: base.valueWeights,
          epsilon: 0,
          configuration: base.configuration))
      }
      var nonFinite = base.input.storage
      nonFinite[1] = .infinity
      passed += expectError(name: "reject non-finite input", failures: &failures) {
        _ = try implementation(FusedQKVRequest(
          input: FloatTensor(nonFinite, shape: base.input.shape),
          gamma: base.gamma,
          queryWeights: base.queryWeights,
          keyWeights: base.keyWeights,
          valueWeights: base.valueWeights,
          epsilon: base.epsilon,
          configuration: base.configuration))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge setup", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 10, failures: failures)
  }

  private static func valueCases() throws -> [(String, FusedQKVRequest)] {
    let small = try smallConfiguration()
    let maximum = try DecoderConfiguration(
      modelDimension: 256,
      hiddenDimension: 8,
      queryHeadCount: 4,
      keyValueHeadCount: 4,
      headDimension: 64,
      rotaryDimension: 64,
      rmsNormEpsilon: 1e-5)
    return [
      ("two-token GQA projection", try makeRequest(sequenceLength: 2, configuration: small)),
      ("sequence-length boundary", try makeRequest(
        sequenceLength: P043FusedQKVContract.maximumSequenceLength,
        configuration: small)),
      ("dimension and projection boundary", try makeRequest(
        sequenceLength: 1, configuration: maximum)),
      ("nonuniform gamma", try makeRequest(sequenceLength: 3, configuration: small, salt: 11)),
    ]
  }

  private static func smallConfiguration() throws -> DecoderConfiguration {
    try DecoderConfiguration(
      modelDimension: 4,
      hiddenDimension: 6,
      queryHeadCount: 2,
      keyValueHeadCount: 1,
      headDimension: 2,
      rotaryDimension: 2,
      rmsNormEpsilon: 1e-5,
      ropeBase: 100)
  }

  private static func makeRequest(
    sequenceLength: Int,
    configuration: DecoderConfiguration,
    salt: Int = 3
  ) throws -> FusedQKVRequest {
    let model = configuration.modelDimension
    let query = configuration.queryProjectionDimension
    let keyValue = configuration.keyValueProjectionDimension
    func values(count: Int, multiplier: Int, divisor: Float) -> [Float] {
      (0..<count).map { index in
        Float(((index * multiplier + salt) % 29) - 14) / divisor
      }
    }
    return FusedQKVRequest(
      input: try FloatTensor(
        values(count: sequenceLength * model, multiplier: 7, divisor: 9),
        shape: [sequenceLength, model]),
      gamma: try FloatTensor(
        (0..<model).map { 0.65 + Float(($0 * 5 + salt) % 13) / 17 },
        shape: [model]),
      queryWeights: try FloatTensor(
        values(count: query * model, multiplier: 11, divisor: 23),
        shape: [query, model]),
      keyWeights: try FloatTensor(
        values(count: keyValue * model, multiplier: 13, divisor: 19),
        shape: [keyValue, model]),
      valueWeights: try FloatTensor(
        values(count: keyValue * model, multiplier: 17, divisor: 21),
        shape: [keyValue, model]),
      epsilon: configuration.rmsNormEpsilon,
      configuration: configuration)
  }

  private static func oracle(_ request: FusedQKVRequest) throws -> FusedQKVResult {
    try P043FusedQKVContract.validate(request)
    let sequence = request.input.shape[0]
    let model = request.configuration.modelDimension
    func project(_ weights: FloatTensor, outputWidth: Int, heads: Int) throws -> FloatTensor {
      var output = Array(repeating: Float.zero, count: sequence * outputWidth)
      for token in 0..<sequence {
        let meanSquare = (0..<model).reduce(0.0) { partial, feature in
          let value = Double(request.input.storage[token * model + feature])
          return partial + value * value
        } / Double(model)
        let inverseRMS = 1 / sqrt(meanSquare + Double(request.epsilon))
        for channel in 0..<outputWidth {
          let sum = (0..<model).reduce(0.0) { partial, feature in
            let normalized = Double(request.input.storage[token * model + feature])
              * inverseRMS * Double(request.gamma.storage[feature])
            return partial
              + normalized * Double(weights.storage[channel * model + feature])
          }
          output[token * outputWidth + channel] = Float(sum)
        }
      }
      return try FloatTensor(output, shape: [sequence, heads, request.configuration.headDimension])
    }
    return FusedQKVResult(
      queries: try project(
        request.queryWeights,
        outputWidth: request.configuration.queryProjectionDimension,
        heads: request.configuration.queryHeadCount),
      keys: try project(
        request.keyWeights,
        outputWidth: request.configuration.keyValueProjectionDimension,
        heads: request.configuration.keyValueHeadCount),
      values: try project(
        request.valueWeights,
        outputWidth: request.configuration.keyValueProjectionDimension,
        heads: request.configuration.keyValueHeadCount))
  }

  private static func equal(_ lhs: FusedQKVResult, _ rhs: FusedQKVResult) -> Bool {
    equal(lhs.queries, rhs.queries) && equal(lhs.keys, rhs.keys) && equal(lhs.values, rhs.values)
  }

  private static func equal(_ lhs: FloatTensor, _ rhs: FloatTensor) -> Bool {
    lhs.shape == rhs.shape && zip(lhs.storage, rhs.storage).allSatisfy { actual, expected in
      abs(actual - expected)
        <= absoluteTolerance + relativeTolerance * max(abs(actual), abs(expected))
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
        caseName: name, message: "expected an error, but the implementation returned"))
      return 0
    } catch {
      return 1
    }
  }
}