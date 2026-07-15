import Foundation

public enum DecoderBlockError: Error, Equatable, LocalizedError {
  case invalidModelDimension(Int)
  case invalidHiddenDimension(Int)
  case modelDimensionDoesNotMatchHeads(model: Int, projected: Int)
  case invalidRotaryDimension(Int)
  case rotaryDimensionExceedsHeadDimension(rotary: Int, head: Int)
  case invalidRMSNormEpsilon(Float)
  case invalidRoPEBase(Float)
  case emptySequence
  case invalidPositionOffset(Int)
  case positionOverflow
  case shapeMismatch(tensor: String, expected: [Int], actual: [Int])
  case nonFiniteValue(tensor: String, linearIndex: Int)

  public var errorDescription: String? {
    switch self {
    case .invalidModelDimension(let value):
      "Model dimension must be positive; received \(value)."
    case .invalidHiddenDimension(let value):
      "MLP hidden dimension must be positive; received \(value)."
    case .modelDimensionDoesNotMatchHeads(let model, let projected):
      "Model dimension \(model) must equal queryHeadCount * headDimension (\(projected))."
    case .invalidRotaryDimension(let value):
      "Rotary dimension must be positive and even; received \(value)."
    case .rotaryDimensionExceedsHeadDimension(let rotary, let head):
      "Rotary dimension \(rotary) exceeds head dimension \(head)."
    case .invalidRMSNormEpsilon(let value):
      "RMSNorm epsilon must be finite and positive; received \(value)."
    case .invalidRoPEBase(let value):
      "RoPE base must be finite and greater than one; received \(value)."
    case .emptySequence:
      "A decoder block requires at least one token."
    case .invalidPositionOffset(let value):
      "Position offset must be nonnegative; received \(value)."
    case .positionOverflow:
      "Position offset plus sequence length exceeds Int.max."
    case .shapeMismatch(let tensor, let expected, let actual):
      "\(tensor) must have shape \(expected); received \(actual)."
    case .nonFiniteValue(let tensor, let linearIndex):
      "\(tensor) contains a non-finite value at linear index \(linearIndex)."
    }
  }
}

public struct DecoderConfiguration: Sendable, Equatable {
  public let modelDimension: Int
  public let hiddenDimension: Int
  public let queryHeadCount: Int
  public let keyValueHeadCount: Int
  public let headDimension: Int
  public let rotaryDimension: Int
  public let rmsNormEpsilon: Float
  public let ropeBase: Float

  public var queryProjectionDimension: Int { queryHeadCount * headDimension }
  public var keyValueProjectionDimension: Int { keyValueHeadCount * headDimension }
  public func attentionConfiguration(
    queryPositionOffset: Int = 0,
    keyPositionOffset: Int = 0
  ) throws -> AttentionConfiguration {
    try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: keyValueHeadCount,
      headDimension: headDimension,
      queryPositionOffset: queryPositionOffset,
      keyPositionOffset: keyPositionOffset)
  }

  public init(
    modelDimension: Int,
    hiddenDimension: Int,
    queryHeadCount: Int,
    keyValueHeadCount: Int,
    headDimension: Int,
    rotaryDimension: Int,
    rmsNormEpsilon: Float,
    ropeBase: Float = 10_000
  ) throws {
    guard modelDimension > 0 else {
      throw DecoderBlockError.invalidModelDimension(modelDimension)
    }
    guard hiddenDimension > 0 else {
      throw DecoderBlockError.invalidHiddenDimension(hiddenDimension)
    }
    _ = try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: keyValueHeadCount,
      headDimension: headDimension)
    let (projected, overflow) = queryHeadCount.multipliedReportingOverflow(by: headDimension)
    guard !overflow, projected == modelDimension else {
      throw DecoderBlockError.modelDimensionDoesNotMatchHeads(
        model: modelDimension, projected: overflow ? -1 : projected)
    }
    guard rotaryDimension > 0, rotaryDimension.isMultiple(of: 2) else {
      throw DecoderBlockError.invalidRotaryDimension(rotaryDimension)
    }
    guard rotaryDimension <= headDimension else {
      throw DecoderBlockError.rotaryDimensionExceedsHeadDimension(
        rotary: rotaryDimension, head: headDimension)
    }
    guard rmsNormEpsilon.isFinite, rmsNormEpsilon > 0 else {
      throw DecoderBlockError.invalidRMSNormEpsilon(rmsNormEpsilon)
    }
    guard ropeBase.isFinite, ropeBase > 1 else {
      throw DecoderBlockError.invalidRoPEBase(ropeBase)
    }
    self.modelDimension = modelDimension
    self.hiddenDimension = hiddenDimension
    self.queryHeadCount = queryHeadCount
    self.keyValueHeadCount = keyValueHeadCount
    self.headDimension = headDimension
    self.rotaryDimension = rotaryDimension
    self.rmsNormEpsilon = rmsNormEpsilon
    self.ropeBase = ropeBase
  }
}

public struct DecoderBlockWeights: Sendable, Equatable {
  public let attentionNormGamma: FloatTensor
  public let queryWeights: FloatTensor
  public let keyWeights: FloatTensor
  public let valueWeights: FloatTensor
  public let attentionOutputWeights: FloatTensor
  public let mlpNormGamma: FloatTensor
  public let gateWeights: FloatTensor
  public let upWeights: FloatTensor
  public let downWeights: FloatTensor

  public init(
    attentionNormGamma: FloatTensor,
    queryWeights: FloatTensor,
    keyWeights: FloatTensor,
    valueWeights: FloatTensor,
    attentionOutputWeights: FloatTensor,
    mlpNormGamma: FloatTensor,
    gateWeights: FloatTensor,
    upWeights: FloatTensor,
    downWeights: FloatTensor
  ) {
    self.attentionNormGamma = attentionNormGamma
    self.queryWeights = queryWeights
    self.keyWeights = keyWeights
    self.valueWeights = valueWeights
    self.attentionOutputWeights = attentionOutputWeights
    self.mlpNormGamma = mlpNormGamma
    self.gateWeights = gateWeights
    self.upWeights = upWeights
    self.downWeights = downWeights
  }
}

public struct DecoderBlockState: Sendable, Equatable {
  public let residual: FloatTensor
  public let positionOffset: Int

  public init(residual: FloatTensor, positionOffset: Int) {
    self.residual = residual
    self.positionOffset = positionOffset
  }
}

public struct DecoderBlockIntermediates: Sendable, Equatable {
  public let attentionNormalized: FloatTensor
  public let queries: FloatTensor
  public let keys: FloatTensor
  public let values: FloatTensor
  public let rotatedQueries: FloatTensor
  public let rotatedKeys: FloatTensor
  public let attentionHeads: FloatTensor
  public let concatenatedAttention: FloatTensor
  public let attentionProjection: FloatTensor
  public let postAttentionResidual: FloatTensor
  public let mlpNormalized: FloatTensor
  public let gateProjection: FloatTensor
  public let upProjection: FloatTensor
  public let activatedGate: FloatTensor
  public let gatedHidden: FloatTensor
  public let downProjection: FloatTensor

  public init(
    attentionNormalized: FloatTensor,
    queries: FloatTensor,
    keys: FloatTensor,
    values: FloatTensor,
    rotatedQueries: FloatTensor,
    rotatedKeys: FloatTensor,
    attentionHeads: FloatTensor,
    concatenatedAttention: FloatTensor,
    attentionProjection: FloatTensor,
    postAttentionResidual: FloatTensor,
    mlpNormalized: FloatTensor,
    gateProjection: FloatTensor,
    upProjection: FloatTensor,
    activatedGate: FloatTensor,
    gatedHidden: FloatTensor,
    downProjection: FloatTensor
  ) {
    self.attentionNormalized = attentionNormalized
    self.queries = queries
    self.keys = keys
    self.values = values
    self.rotatedQueries = rotatedQueries
    self.rotatedKeys = rotatedKeys
    self.attentionHeads = attentionHeads
    self.concatenatedAttention = concatenatedAttention
    self.attentionProjection = attentionProjection
    self.postAttentionResidual = postAttentionResidual
    self.mlpNormalized = mlpNormalized
    self.gateProjection = gateProjection
    self.upProjection = upProjection
    self.activatedGate = activatedGate
    self.gatedHidden = gatedHidden
    self.downProjection = downProjection
  }
}

public struct DecoderBlockResult: Sendable, Equatable {
  public let state: DecoderBlockState
  public let intermediates: DecoderBlockIntermediates

  public init(state: DecoderBlockState, intermediates: DecoderBlockIntermediates) {
    self.state = state
    self.intermediates = intermediates
  }
}

public typealias DecoderBlockImplementation = (
  _ state: DecoderBlockState,
  _ weights: DecoderBlockWeights,
  _ configuration: DecoderConfiguration
) throws -> DecoderBlockResult

public enum P035DecoderBlockContract {
  public static func validate(
    state: DecoderBlockState,
    weights: DecoderBlockWeights,
    configuration: DecoderConfiguration
  ) throws {
    guard state.residual.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: state.residual.rank)
    }
    guard state.residual.shape[0] > 0 else { throw DecoderBlockError.emptySequence }
    guard state.residual.shape[1] == configuration.modelDimension else {
      throw DecoderBlockError.shapeMismatch(
        tensor: "Residual state",
        expected: [state.residual.shape[0], configuration.modelDimension],
        actual: state.residual.shape)
    }
    guard state.positionOffset >= 0 else {
      throw DecoderBlockError.invalidPositionOffset(state.positionOffset)
    }
    let (_, overflow) = state.positionOffset.addingReportingOverflow(state.residual.shape[0] - 1)
    guard !overflow else { throw DecoderBlockError.positionOverflow }
    try validateWeights(weights, configuration: configuration)
    try validateFinite(name: "Residual state", tensor: state.residual)
  }

  public static func validateWeights(
    _ weights: DecoderBlockWeights,
    configuration: DecoderConfiguration
  ) throws {
    let model = configuration.modelDimension
    let hidden = configuration.hiddenDimension
    let query = configuration.queryProjectionDimension
    let keyValue = configuration.keyValueProjectionDimension
    let expected: [(String, FloatTensor, [Int])] = [
      ("Attention RMSNorm gamma", weights.attentionNormGamma, [model]),
      ("Query weights", weights.queryWeights, [query, model]),
      ("Key weights", weights.keyWeights, [keyValue, model]),
      ("Value weights", weights.valueWeights, [keyValue, model]),
      ("Attention output weights", weights.attentionOutputWeights, [model, query]),
      ("MLP RMSNorm gamma", weights.mlpNormGamma, [model]),
      ("Gate weights", weights.gateWeights, [hidden, model]),
      ("Up weights", weights.upWeights, [hidden, model]),
      ("Down weights", weights.downWeights, [model, hidden]),
    ]
    for (name, tensor, shape) in expected {
      guard tensor.shape == shape else {
        throw DecoderBlockError.shapeMismatch(tensor: name, expected: shape, actual: tensor.shape)
      }
      try validateFinite(name: name, tensor: tensor)
    }
  }

  private static func validateFinite(name: String, tensor: FloatTensor) throws {
    if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
      throw DecoderBlockError.nonFiniteValue(tensor: name, linearIndex: index)
    }
  }
}

public enum P035DecoderBlockJudge {
  public static let absoluteTolerance: Float = 4e-5
  public static let relativeTolerance: Float = 8e-5

  public static func evaluate(_ implementation: DecoderBlockImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      let configuration = try fixtureConfiguration()
      let weights = try fixtureWeights(configuration: configuration)
      let states = [
        DecoderBlockState(
          residual: try FloatTensor(
            [0.75, -1.0, 0.5, 1.25, -0.25, 0.4, 1.1, -0.8], shape: [2, 4]),
          positionOffset: 3),
        DecoderBlockState(
          residual: try FloatTensor(
            [0.2, -0.7, 1.3, 0.1, 0.9, 0.25, -0.4, 1.1, -0.6, 0.8, 0.3, -1.2],
            shape: [3, 4]),
          positionOffset: 0),
      ]
      for (index, state) in states.enumerated() {
        let actual = try implementation(state, weights, configuration)
        let expected = try oracle(state, weights, configuration)
        if equal(actual, expected) {
          passed += 1
        } else {
          failures.append(JudgeFailure(
            caseName: index == 0 ? "GQA block with nonzero RoPE offset" : "three-token causal block",
            message: "one or more captured stages differ from the independent ordered oracle"))
        }
      }

      passed += expectError(name: "reject empty sequence", failures: &failures) {
        _ = try implementation(
          DecoderBlockState(
            residual: FloatTensor([], shape: [0, configuration.modelDimension]),
            positionOffset: 0),
          weights,
          configuration)
      }
      passed += expectError(name: "reject residual width", failures: &failures) {
        _ = try implementation(
          DecoderBlockState(
            residual: FloatTensor([1, 2, 3], shape: [1, 3]), positionOffset: 0),
          weights,
          configuration)
      }
      passed += expectError(name: "reject negative position", failures: &failures) {
        _ = try implementation(
          DecoderBlockState(
            residual: FloatTensor([1, 2, 3, 4], shape: [1, 4]), positionOffset: -1),
          weights,
          configuration)
      }
      var wrongWeights = weights
      wrongWeights = DecoderBlockWeights(
        attentionNormGamma: weights.attentionNormGamma,
        queryWeights: try FloatTensor(Array(repeating: 0, count: 12), shape: [3, 4]),
        keyWeights: weights.keyWeights,
        valueWeights: weights.valueWeights,
        attentionOutputWeights: weights.attentionOutputWeights,
        mlpNormGamma: weights.mlpNormGamma,
        gateWeights: weights.gateWeights,
        upWeights: weights.upWeights,
        downWeights: weights.downWeights)
      passed += expectError(name: "reject wrong query weight shape", failures: &failures) {
        _ = try implementation(
          DecoderBlockState(
            residual: FloatTensor([1, 2, 3, 4], shape: [1, 4]), positionOffset: 0),
          wrongWeights,
          configuration)
      }
      passed += expectError(name: "reject non-finite input", failures: &failures) {
        _ = try implementation(
          DecoderBlockState(
            residual: FloatTensor([1, .infinity, 3, 4], shape: [1, 4]), positionOffset: 0),
          weights,
          configuration)
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 7, failures: failures)
  }

  private static func fixtureConfiguration() throws -> DecoderConfiguration {
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

  private static func fixtureWeights(
    configuration: DecoderConfiguration
  ) throws -> DecoderBlockWeights {
    func matrix(_ rows: Int, _ columns: Int, salt: Int) throws -> FloatTensor {
      let values = (0..<(rows * columns)).map { index in
        Float(((index * 7 + salt * 3) % 19) - 9) / 13
      }
      return try FloatTensor(values, shape: [rows, columns])
    }
    return DecoderBlockWeights(
      attentionNormGamma: try FloatTensor([1.1, 0.8, 1.25, 0.9], shape: [4]),
      queryWeights: try matrix(4, 4, salt: 1),
      keyWeights: try matrix(2, 4, salt: 2),
      valueWeights: try matrix(2, 4, salt: 3),
      attentionOutputWeights: try matrix(4, 4, salt: 4),
      mlpNormGamma: try FloatTensor([0.95, 1.2, 0.75, 1.05], shape: [4]),
      gateWeights: try matrix(6, 4, salt: 5),
      upWeights: try matrix(6, 4, salt: 6),
      downWeights: try matrix(4, 6, salt: 7))
  }

  private static func oracle(
    _ state: DecoderBlockState,
    _ weights: DecoderBlockWeights,
    _ configuration: DecoderConfiguration
  ) throws -> DecoderBlockResult {
    try P035DecoderBlockContract.validate(
      state: state, weights: weights, configuration: configuration)
    let sequence = state.residual.shape[0]
    let model = configuration.modelDimension
    let hidden = configuration.hiddenDimension

    func tensor(_ values: [Double], _ shape: [Int]) throws -> FloatTensor {
      try FloatTensor(values.map(Float.init), shape: shape)
    }
    func normalize(_ input: FloatTensor, _ gamma: FloatTensor) throws -> FloatTensor {
      var output = Array(repeating: 0.0, count: input.elementCount)
      for row in 0..<input.shape[0] {
        let meanSquare = (0..<input.shape[1]).reduce(0.0) { sum, column in
          let value = Double(input.storage[row * input.shape[1] + column])
          return sum + value * value
        } / Double(input.shape[1])
        let inverse = 1 / sqrt(meanSquare + Double(configuration.rmsNormEpsilon))
        for column in 0..<input.shape[1] {
          output[row * input.shape[1] + column] =
            Double(input.storage[row * input.shape[1] + column]) * inverse
            * Double(gamma.storage[column])
        }
      }
      return try tensor(output, input.shape)
    }
    func project(_ input: FloatTensor, _ weight: FloatTensor) throws -> FloatTensor {
      let rows = input.shape[0]
      let inputWidth = input.shape[1]
      let outputWidth = weight.shape[0]
      var output = Array(repeating: 0.0, count: rows * outputWidth)
      for row in 0..<rows {
        for outputChannel in 0..<outputWidth {
          output[row * outputWidth + outputChannel] = (0..<inputWidth).reduce(0.0) {
            $0 + Double(input.storage[row * inputWidth + $1])
              * Double(weight.storage[outputChannel * inputWidth + $1])
          }
        }
      }
      return try tensor(output, [rows, outputWidth])
    }
    func reshapeHeads(_ input: FloatTensor, heads: Int) throws -> FloatTensor {
      try FloatTensor(
        input.storage, shape: [sequence, heads, configuration.headDimension])
    }
    func rotate(_ input: FloatTensor) throws -> FloatTensor {
      var output = input.storage.map(Double.init)
      for token in 0..<sequence {
        let position = Double(state.positionOffset + token)
        for head in 0..<input.shape[1] {
          let start = (token * input.shape[1] + head) * configuration.headDimension
          for pairStart in stride(from: 0, to: configuration.rotaryDimension, by: 2) {
            let pair = pairStart / 2
            let angle = position / pow(
              Double(configuration.ropeBase),
              Double(2 * pair) / Double(configuration.rotaryDimension))
            let first = Double(input.storage[start + pairStart])
            let second = Double(input.storage[start + pairStart + 1])
            output[start + pairStart] = first * cos(angle) - second * sin(angle)
            output[start + pairStart + 1] = first * sin(angle) + second * cos(angle)
          }
        }
      }
      return try tensor(output, input.shape)
    }
    func attention(_ queries: FloatTensor, _ keys: FloatTensor, _ values: FloatTensor) throws
      -> FloatTensor
    {
      let head = configuration.headDimension
      let group = configuration.queryHeadCount / configuration.keyValueHeadCount
      let scale = 1 / sqrt(Double(head))
      var output = Array(repeating: 0.0, count: queries.elementCount)
      for query in 0..<sequence {
        for queryHead in 0..<configuration.queryHeadCount {
          let keyValueHead = queryHead / group
          let scores = (0...query).map { key -> Double in
            (0..<head).reduce(0.0) { partial, feature in
              let queryIndex = (query * configuration.queryHeadCount + queryHead) * head + feature
              let keyIndex = (key * configuration.keyValueHeadCount + keyValueHead) * head + feature
              return partial + Double(queries.storage[queryIndex]) * Double(keys.storage[keyIndex])
            } * scale
          }
          let maximum = scores.max()!
          let exponentials = scores.map { exp($0 - maximum) }
          let denominator = exponentials.reduce(0, +)
          for feature in 0..<head {
            let outputIndex = (query * configuration.queryHeadCount + queryHead) * head + feature
            output[outputIndex] = (0...query).reduce(0.0) { partial, key in
              let valueIndex =
                (key * configuration.keyValueHeadCount + keyValueHead) * head + feature
              return partial + exponentials[key] / denominator * Double(values.storage[valueIndex])
            }
          }
        }
      }
      return try tensor(output, queries.shape)
    }
    func add(_ lhs: FloatTensor, _ rhs: FloatTensor) throws -> FloatTensor {
      try tensor(zip(lhs.storage, rhs.storage).map { Double($0) + Double($1) }, lhs.shape)
    }

    let attentionNormalized = try normalize(state.residual, weights.attentionNormGamma)
    let queryProjection = try project(attentionNormalized, weights.queryWeights)
    let keyProjection = try project(attentionNormalized, weights.keyWeights)
    let valueProjection = try project(attentionNormalized, weights.valueWeights)
    let queries = try reshapeHeads(queryProjection, heads: configuration.queryHeadCount)
    let keys = try reshapeHeads(keyProjection, heads: configuration.keyValueHeadCount)
    let values = try reshapeHeads(valueProjection, heads: configuration.keyValueHeadCount)
    let rotatedQueries = try rotate(queries)
    let rotatedKeys = try rotate(keys)
    let attentionHeads = try attention(rotatedQueries, rotatedKeys, values)
    let concatenatedAttention = try FloatTensor(
      attentionHeads.storage, shape: [sequence, model])
    let attentionProjection = try project(
      concatenatedAttention, weights.attentionOutputWeights)
    let postAttentionResidual = try add(state.residual, attentionProjection)
    let mlpNormalized = try normalize(postAttentionResidual, weights.mlpNormGamma)
    let gateProjection = try project(mlpNormalized, weights.gateWeights)
    let upProjection = try project(mlpNormalized, weights.upWeights)
    let activatedGate = try tensor(gateProjection.storage.map {
      let value = Double($0)
      return value / (1 + exp(-value))
    }, [sequence, hidden])
    let gatedHidden = try tensor(
      zip(activatedGate.storage, upProjection.storage).map { Double($0) * Double($1) },
      [sequence, hidden])
    let downProjection = try project(gatedHidden, weights.downWeights)
    let finalResidual = try add(postAttentionResidual, downProjection)
    let intermediates = DecoderBlockIntermediates(
      attentionNormalized: attentionNormalized,
      queries: queries,
      keys: keys,
      values: values,
      rotatedQueries: rotatedQueries,
      rotatedKeys: rotatedKeys,
      attentionHeads: attentionHeads,
      concatenatedAttention: concatenatedAttention,
      attentionProjection: attentionProjection,
      postAttentionResidual: postAttentionResidual,
      mlpNormalized: mlpNormalized,
      gateProjection: gateProjection,
      upProjection: upProjection,
      activatedGate: activatedGate,
      gatedHidden: gatedHidden,
      downProjection: downProjection)
    return DecoderBlockResult(
      state: DecoderBlockState(residual: finalResidual, positionOffset: state.positionOffset),
      intermediates: intermediates)
  }

  private static func equal(_ lhs: DecoderBlockResult, _ rhs: DecoderBlockResult) -> Bool {
    lhs.state.positionOffset == rhs.state.positionOffset
      && equal(lhs.state.residual, rhs.state.residual)
      && equal(lhs.intermediates.attentionNormalized, rhs.intermediates.attentionNormalized)
      && equal(lhs.intermediates.queries, rhs.intermediates.queries)
      && equal(lhs.intermediates.keys, rhs.intermediates.keys)
      && equal(lhs.intermediates.values, rhs.intermediates.values)
      && equal(lhs.intermediates.rotatedQueries, rhs.intermediates.rotatedQueries)
      && equal(lhs.intermediates.rotatedKeys, rhs.intermediates.rotatedKeys)
      && equal(lhs.intermediates.attentionHeads, rhs.intermediates.attentionHeads)
      && equal(lhs.intermediates.concatenatedAttention, rhs.intermediates.concatenatedAttention)
      && equal(lhs.intermediates.attentionProjection, rhs.intermediates.attentionProjection)
      && equal(lhs.intermediates.postAttentionResidual, rhs.intermediates.postAttentionResidual)
      && equal(lhs.intermediates.mlpNormalized, rhs.intermediates.mlpNormalized)
      && equal(lhs.intermediates.gateProjection, rhs.intermediates.gateProjection)
      && equal(lhs.intermediates.upProjection, rhs.intermediates.upProjection)
      && equal(lhs.intermediates.activatedGate, rhs.intermediates.activatedGate)
      && equal(lhs.intermediates.gatedHidden, rhs.intermediates.gatedHidden)
      && equal(lhs.intermediates.downProjection, rhs.intermediates.downProjection)
  }

  private static func equal(_ lhs: FloatTensor, _ rhs: FloatTensor) -> Bool {
    lhs.shape == rhs.shape && zip(lhs.storage, rhs.storage).allSatisfy { actual, expected in
      abs(actual - expected) <= absoluteTolerance + relativeTolerance * abs(expected)
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