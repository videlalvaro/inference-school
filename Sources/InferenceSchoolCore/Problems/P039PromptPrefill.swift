import Foundation

public struct PromptPrefillRequest: Sendable, Equatable {
  public let model: MiniDecoderModel
  public let tokenIDs: [Int]
  public let positionOffset: Int

  public init(model: MiniDecoderModel, tokenIDs: [Int], positionOffset: Int = 0) {
    self.model = model
    self.tokenIDs = tokenIDs
    self.positionOffset = positionOffset
  }
}

public struct PromptPrefillResult: Sendable, Equatable {
  public let promptTokenCount: Int
  public let finalResidual: FloatTensor
  public let finalNormalized: FloatTensor
  public let finalHidden: FloatTensor
  public let logits: FloatTensor
  public let layers: [MiniDecoderLayerTrace]
  public let cacheCounts: [Int]
  public let cachePositions: [[Int]]
  public let work: MiniDecoderWorkModel

  public init(
    promptTokenCount: Int,
    finalResidual: FloatTensor,
    finalNormalized: FloatTensor,
    finalHidden: FloatTensor,
    logits: FloatTensor,
    layers: [MiniDecoderLayerTrace],
    cacheCounts: [Int],
    cachePositions: [[Int]],
    work: MiniDecoderWorkModel
  ) {
    self.promptTokenCount = promptTokenCount
    self.finalResidual = finalResidual
    self.finalNormalized = finalNormalized
    self.finalHidden = finalHidden
    self.logits = logits
    self.layers = layers
    self.cacheCounts = cacheCounts
    self.cachePositions = cachePositions
    self.work = work
  }
}

public typealias PromptPrefillImplementation = (
  _ request: PromptPrefillRequest,
  _ cache: ContiguousKVCache
) throws -> PromptPrefillResult

public enum P039PromptPrefillContract {
  public static func validate(
    request: PromptPrefillRequest,
    cache: any KVCacheReadable
  ) throws {
    try request.model.validate(tokenIDs: request.tokenIDs)
    try request.model.validate(cache: cache)
    guard request.positionOffset >= 0 else {
      throw DecoderBlockError.invalidPositionOffset(request.positionOffset)
    }
    let (_, overflow) = request.positionOffset.addingReportingOverflow(
      request.tokenIDs.count - 1)
    guard !overflow else { throw DecoderBlockError.positionOverflow }
    guard cache.configuration.capacity >= request.tokenIDs.count else {
      throw KVCacheError.capacityExceeded(
        layer: 0, capacity: cache.configuration.capacity)
    }
    for layer in 0..<request.model.layerCount {
      let count = try cache.count(layer: layer)
      guard count == 0 else {
        throw MiniDecoderError.cacheNotEmpty(layer: layer, count: count)
      }
    }
  }
}

enum MiniDecoderReference {
  static let absoluteTolerance: Float = 6e-5
  static let relativeTolerance: Float = 1.2e-4

  static func prefill(_ request: PromptPrefillRequest) throws -> PromptPrefillResult {
    let cache = ContiguousKVCache(
      configuration: try request.model.cacheConfiguration(capacity: request.tokenIDs.count))
    try P039PromptPrefillContract.validate(request: request, cache: cache)
    let model = request.model
    let sequence = request.tokenIDs.count
    let dimension = model.configuration.modelDimension
    var embedded: [Float] = []
    embedded.reserveCapacity(sequence * dimension)
    for tokenID in request.tokenIDs {
      let start = tokenID * dimension
      embedded.append(contentsOf: model.tokenEmbedding.storage[start..<(start + dimension)])
    }
    var residual = try FloatTensor(embedded, shape: [sequence, dimension])
    var traces: [MiniDecoderLayerTrace] = []
    for (layer, weights) in model.blocks.enumerated() {
      let input = residual
      let result = try applyBlock(
        residual: input,
        positionOffset: request.positionOffset,
        weights: weights,
        configuration: model.configuration)
      for token in 0..<sequence {
        let key = try tokenSlice(
          result.intermediates.rotatedKeys,
          token: token,
          headCount: model.configuration.keyValueHeadCount,
          headDimension: model.configuration.headDimension)
        let value = try tokenSlice(
          result.intermediates.values,
          token: token,
          headCount: model.configuration.keyValueHeadCount,
          headDimension: model.configuration.headDimension)
        try cache.append(
          layer: layer,
          logicalPosition: request.positionOffset + token,
          key: key,
          value: value)
      }
      residual = result.state.residual
      traces.append(MiniDecoderLayerTrace(
        layerIndex: layer,
        residualInput: input,
        block: result,
        cachePositions: try cache.logicalPositions(layer: layer)))
    }
    let normalized = try normalize(
      residual, gamma: model.finalNormGamma, epsilon: model.configuration.rmsNormEpsilon)
    let lastStart = (sequence - 1) * dimension
    let finalHidden = try FloatTensor(
      Array(normalized.storage[lastStart..<(lastStart + dimension)]), shape: [dimension])
    let logits = try projectVector(finalHidden, weights: model.outputWeights)
    let counts = try (0..<model.layerCount).map { try cache.count(layer: $0) }
    let positions = try (0..<model.layerCount).map {
      try cache.logicalPositions(layer: $0)
    }
    return PromptPrefillResult(
      promptTokenCount: sequence,
      finalResidual: residual,
      finalNormalized: normalized,
      finalHidden: finalHidden,
      logits: logits,
      layers: traces,
      cacheCounts: counts,
      cachePositions: positions,
      work: try workModel(model: model, tokenCount: sequence, decode: false))
  }

  static func applyBlock(
    residual: FloatTensor,
    positionOffset: Int,
    weights: DecoderBlockWeights,
    configuration: DecoderConfiguration,
    ropePositionOffsetDelta: Int = 0,
    additiveNormGamma: Bool = false
  ) throws -> DecoderBlockResult {
    let sequence = residual.shape[0]
    let model = configuration.modelDimension
    let hidden = configuration.hiddenDimension
    let attentionNormalized = try normalize(
      residual,
      gamma: weights.attentionNormGamma,
      epsilon: configuration.rmsNormEpsilon,
      additiveGamma: additiveNormGamma)
    let queryProjection = try project(attentionNormalized, weights: weights.queryWeights)
    let keyProjection = try project(attentionNormalized, weights: weights.keyWeights)
    let valueProjection = try project(attentionNormalized, weights: weights.valueWeights)
    let queries = try FloatTensor(
      queryProjection.storage,
      shape: [sequence, configuration.queryHeadCount, configuration.headDimension])
    let keys = try FloatTensor(
      keyProjection.storage,
      shape: [sequence, configuration.keyValueHeadCount, configuration.headDimension])
    let values = try FloatTensor(
      valueProjection.storage,
      shape: [sequence, configuration.keyValueHeadCount, configuration.headDimension])
    let rotatedQueries = try rotate(
      queries,
      positionOffset: positionOffset + ropePositionOffsetDelta,
      configuration: configuration)
    let rotatedKeys = try rotate(
      keys,
      positionOffset: positionOffset + ropePositionOffsetDelta,
      configuration: configuration)
    let attentionHeads = try attention(
      queries: rotatedQueries,
      keys: rotatedKeys,
      values: values,
      configuration: configuration)
    let concatenatedAttention = try FloatTensor(
      attentionHeads.storage, shape: [sequence, model])
    let attentionProjection = try project(
      concatenatedAttention, weights: weights.attentionOutputWeights)
    let postAttentionResidual = try add(residual, attentionProjection)
    let mlpNormalized = try normalize(
      postAttentionResidual,
      gamma: weights.mlpNormGamma,
      epsilon: configuration.rmsNormEpsilon,
      additiveGamma: additiveNormGamma)
    let gateProjection = try project(mlpNormalized, weights: weights.gateWeights)
    let upProjection = try project(mlpNormalized, weights: weights.upWeights)
    let activatedGate = try FloatTensor(gateProjection.storage.map { value in
      let double = Double(value)
      return Float(double / (1 + exp(-double)))
    }, shape: [sequence, hidden])
    let gatedHidden = try FloatTensor(
      zip(activatedGate.storage, upProjection.storage).map { Float(Double($0) * Double($1)) },
      shape: [sequence, hidden])
    let downProjection = try project(gatedHidden, weights: weights.downWeights)
    let finalResidual = try add(postAttentionResidual, downProjection)
    return DecoderBlockResult(
      state: DecoderBlockState(residual: finalResidual, positionOffset: positionOffset),
      intermediates: DecoderBlockIntermediates(
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
        downProjection: downProjection))
  }

  static func normalize(
    _ input: FloatTensor,
    gamma: FloatTensor,
    epsilon: Float,
    additiveGamma: Bool = false
  ) throws -> FloatTensor {
    let rows = input.rank == 1 ? 1 : input.shape[0]
    let width = input.shape.last!
    var output = Array(repeating: Float.zero, count: input.elementCount)
    for row in 0..<rows {
      var sumSquares = 0.0
      for column in 0..<width {
        let value = Double(input.storage[row * width + column])
        sumSquares += value * value
      }
      let inverse = 1 / sqrt(sumSquares / Double(width) + Double(epsilon))
      for column in 0..<width {
        let scale = Double(gamma.storage[column]) + (additiveGamma ? 1 : 0)
        output[row * width + column] = Float(
          Double(input.storage[row * width + column]) * inverse * scale)
      }
    }
    return try FloatTensor(output, shape: input.shape)
  }

  static func project(_ input: FloatTensor, weights: FloatTensor) throws -> FloatTensor {
    let rows = input.shape[0]
    let inputWidth = input.shape[1]
    let outputWidth = weights.shape[0]
    var output = Array(repeating: Float.zero, count: rows * outputWidth)
    for row in 0..<rows {
      for outputChannel in 0..<outputWidth {
        var sum = 0.0
        for inputChannel in 0..<inputWidth {
          sum += Double(input.storage[row * inputWidth + inputChannel])
            * Double(weights.storage[outputChannel * inputWidth + inputChannel])
        }
        output[row * outputWidth + outputChannel] = Float(sum)
      }
    }
    return try FloatTensor(output, shape: [rows, outputWidth])
  }

  static func projectVector(_ input: FloatTensor, weights: FloatTensor) throws -> FloatTensor {
    let matrix = try FloatTensor(input.storage, shape: [1, input.shape[0]])
    let projected = try project(matrix, weights: weights)
    return try FloatTensor(projected.storage, shape: [weights.shape[0]])
  }

  static func rotate(
    _ input: FloatTensor,
    positionOffset: Int,
    configuration: DecoderConfiguration
  ) throws -> FloatTensor {
    var output = input.storage
    for token in 0..<input.shape[0] {
      let position = Double(positionOffset + token)
      for head in 0..<input.shape[1] {
        let start = (token * input.shape[1] + head) * configuration.headDimension
        for pairStart in stride(from: 0, to: configuration.rotaryDimension, by: 2) {
          let pair = pairStart / 2
          let angle = position / pow(
            Double(configuration.ropeBase),
            Double(2 * pair) / Double(configuration.rotaryDimension))
          let first = Double(input.storage[start + pairStart])
          let second = Double(input.storage[start + pairStart + 1])
          output[start + pairStart] = Float(first * cos(angle) - second * sin(angle))
          output[start + pairStart + 1] = Float(first * sin(angle) + second * cos(angle))
        }
      }
    }
    return try FloatTensor(output, shape: input.shape)
  }

  static func attention(
    queries: FloatTensor,
    keys: FloatTensor,
    values: FloatTensor,
    configuration: DecoderConfiguration
  ) throws -> FloatTensor {
    let sequence = queries.shape[0]
    let headDimension = configuration.headDimension
    let groupSize = configuration.queryHeadCount / configuration.keyValueHeadCount
    let scale = 1 / sqrt(Double(headDimension))
    var output = Array(repeating: Float.zero, count: queries.elementCount)
    for query in 0..<sequence {
      for queryHead in 0..<configuration.queryHeadCount {
        let keyValueHead = queryHead / groupSize
        var scores: [Double] = []
        for key in 0...query {
          var dot = 0.0
          for feature in 0..<headDimension {
            dot += Double(queries.storage[
              (query * configuration.queryHeadCount + queryHead) * headDimension + feature])
              * Double(keys.storage[
                (key * configuration.keyValueHeadCount + keyValueHead) * headDimension + feature])
          }
          scores.append(dot * scale)
        }
        let maximum = scores.max()!
        let exponentials = scores.map { exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        for feature in 0..<headDimension {
          var sum = 0.0
          for key in 0...query {
            sum += exponentials[key] / denominator
              * Double(values.storage[
                (key * configuration.keyValueHeadCount + keyValueHead) * headDimension + feature])
          }
          output[(query * configuration.queryHeadCount + queryHead) * headDimension + feature]
            = Float(sum)
        }
      }
    }
    return try FloatTensor(output, shape: queries.shape)
  }

  static func add(_ lhs: FloatTensor, _ rhs: FloatTensor) throws -> FloatTensor {
    try FloatTensor(
      zip(lhs.storage, rhs.storage).map { Float(Double($0) + Double($1)) },
      shape: lhs.shape)
  }

  static func tokenSlice(
    _ tensor: FloatTensor,
    token: Int,
    headCount: Int,
    headDimension: Int
  ) throws -> FloatTensor {
    let count = headCount * headDimension
    let start = token * count
    return try FloatTensor(
      Array(tensor.storage[start..<(start + count)]),
      shape: [headCount, headDimension])
  }

  static func workModel(
    model: MiniDecoderModel,
    tokenCount: Int,
    attentionTokenCount: Int? = nil,
    decode: Bool
  ) throws -> MiniDecoderWorkModel {
    let configuration = model.configuration
    let dimension = configuration.modelDimension
    let keyValueWidth = configuration.keyValueProjectionDimension
    let hidden = configuration.hiddenDimension
    let layerProjectionTerms = try sumChecked([
      try multiplyChecked(dimension, dimension, "query projection"),
      try multiplyChecked(dimension, keyValueWidth, "key projection"),
      try multiplyChecked(dimension, keyValueWidth, "value projection"),
      try multiplyChecked(dimension, dimension, "attention output projection"),
      try multiplyChecked(dimension, hidden, "gate projection"),
      try multiplyChecked(dimension, hidden, "up projection"),
      try multiplyChecked(hidden, dimension, "down projection"),
    ], "projection terms")
    let projectionFLOPs = try multiplyChecked(
      2,
      try multiplyChecked(
        tokenCount,
        try multiplyChecked(model.layerCount, layerProjectionTerms, "layer projections"),
        "token projections"),
      "projection FLOPs")
    let outputFLOPs = try multiplyChecked(
      2,
      try multiplyChecked(model.vocabularySize, dimension, "output projection"),
      "output FLOPs")
    let totalProjectionFLOPs = try sumChecked(
      [projectionFLOPs, outputFLOPs], "total projection FLOPs")
    let visiblePairs = decode
      ? (attentionTokenCount ?? tokenCount)
      : try multiplyChecked(tokenCount, tokenCount + 1, "causal pair count") / 2
    let attentionFLOPs = try multiplyChecked(
      4,
      try multiplyChecked(
        visiblePairs,
        try multiplyChecked(
          configuration.queryHeadCount,
          try multiplyChecked(configuration.headDimension, model.layerCount, "attention width"),
          "attention heads"),
        "attention pairs"),
      "attention FLOPs")
    let weightElements = model.blocks.reduce(model.outputWeights.elementCount) { partial, block in
      partial + block.queryWeights.elementCount + block.keyWeights.elementCount
        + block.valueWeights.elementCount + block.attentionOutputWeights.elementCount
        + block.gateWeights.elementCount + block.upWeights.elementCount
        + block.downWeights.elementCount + block.attentionNormGamma.elementCount
        + block.mlpNormGamma.elementCount
    } + model.finalNormGamma.elementCount
    let estimatedWeightBytesRead = try multiplyChecked(
      weightElements, MemoryLayout<Float>.stride, "weight bytes")
    let cacheElements = try multiplyChecked(
      tokenCount,
      try multiplyChecked(
        model.layerCount,
        try multiplyChecked(
          2,
          try multiplyChecked(
            configuration.keyValueHeadCount,
            configuration.headDimension,
            "KV vector width"),
          "key and value"),
        "layer cache writes"),
      "token cache writes")
    return MiniDecoderWorkModel(
      projectionFLOPs: totalProjectionFLOPs,
      attentionFLOPs: attentionFLOPs,
      estimatedWeightBytesRead: estimatedWeightBytesRead,
      cacheBytesWritten: try multiplyChecked(
        cacheElements, MemoryLayout<Float>.stride, "cache bytes"),
      keyValueProjectionInputTokens: try multiplyChecked(
        tokenCount, model.layerCount, "KV projection input tokens"),
      priorKeyValueTokensReprojected: 0)
  }

  static func approximatelyEqual(_ lhs: FloatTensor, _ rhs: FloatTensor) -> Bool {
    lhs.shape == rhs.shape && zip(lhs.storage, rhs.storage).allSatisfy { actual, expected in
      abs(actual - expected) <= absoluteTolerance + relativeTolerance * abs(expected)
    }
  }

  static func resultApproximatelyEqual(
    _ lhs: PromptPrefillResult,
    _ rhs: PromptPrefillResult
  ) -> Bool {
    lhs.promptTokenCount == rhs.promptTokenCount
      && approximatelyEqual(lhs.finalResidual, rhs.finalResidual)
      && approximatelyEqual(lhs.finalNormalized, rhs.finalNormalized)
      && approximatelyEqual(lhs.finalHidden, rhs.finalHidden)
      && approximatelyEqual(lhs.logits, rhs.logits)
      && lhs.layers.count == rhs.layers.count
      && zip(lhs.layers, rhs.layers).allSatisfy(layerApproximatelyEqual)
      && lhs.cacheCounts == rhs.cacheCounts
      && lhs.cachePositions == rhs.cachePositions
  }

  static func layerApproximatelyEqual(
    _ lhs: MiniDecoderLayerTrace,
    _ rhs: MiniDecoderLayerTrace
  ) -> Bool {
    let left = lhs.block.intermediates
    let right = rhs.block.intermediates
    return lhs.layerIndex == rhs.layerIndex
      && lhs.cachePositions == rhs.cachePositions
      && approximatelyEqual(lhs.residualInput, rhs.residualInput)
      && approximatelyEqual(lhs.block.state.residual, rhs.block.state.residual)
      && approximatelyEqual(left.attentionNormalized, right.attentionNormalized)
      && approximatelyEqual(left.queries, right.queries)
      && approximatelyEqual(left.keys, right.keys)
      && approximatelyEqual(left.values, right.values)
      && approximatelyEqual(left.rotatedQueries, right.rotatedQueries)
      && approximatelyEqual(left.rotatedKeys, right.rotatedKeys)
      && approximatelyEqual(left.attentionHeads, right.attentionHeads)
      && approximatelyEqual(left.postAttentionResidual, right.postAttentionResidual)
      && approximatelyEqual(left.mlpNormalized, right.mlpNormalized)
      && approximatelyEqual(left.gateProjection, right.gateProjection)
      && approximatelyEqual(left.upProjection, right.upProjection)
      && approximatelyEqual(left.gatedHidden, right.gatedHidden)
      && approximatelyEqual(left.downProjection, right.downProjection)
  }

  private static func multiplyChecked(
    _ lhs: Int,
    _ rhs: Int,
    _ context: String
  ) throws -> Int {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw MiniDecoderError.integerOverflow(context: context) }
    return value
  }

  private static func sumChecked(_ values: [Int], _ context: String) throws -> Int {
    try values.reduce(0) { partial, value in
      let (next, overflow) = partial.addingReportingOverflow(value)
      guard !overflow else { throw MiniDecoderError.integerOverflow(context: context) }
      return next
    }
  }
}

public enum P039PromptPrefillJudge {
  public static func evaluate(_ implementation: PromptPrefillImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      let model = try EducationalMiniModelFixture.make()
      let request = PromptPrefillRequest(
        model: model,
        tokenIDs: EducationalMiniModelFixture.defaultPrompt,
        positionOffset: 5)
      let cache = ContiguousKVCache(
        configuration: try model.cacheConfiguration(capacity: 8))
      let actual = try implementation(request, cache)
      let expected = try MiniDecoderReference.prefill(request)
      if MiniDecoderReference.resultApproximatelyEqual(actual, expected) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "ordered multi-layer prefill and final-token logits",
          message: "residuals, captures, final hidden state, logits, or layer order differ from the independent Double oracle"))
      }
      var cacheMatches = true
      for layer in 0..<model.layerCount {
        let materialized = try cache.materialized(layer: layer)
        cacheMatches = cacheMatches
          && materialized.positions == expected.cachePositions[layer]
          && MiniDecoderReference.approximatelyEqual(
            materialized.keys, expected.layers[layer].block.intermediates.rotatedKeys)
          && MiniDecoderReference.approximatelyEqual(
            materialized.values, expected.layers[layer].block.intermediates.values)
      }
      if cacheMatches {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "cache rotated keys and values for every layer",
          message: "the cache must contain each layer's rotated K and unrotated V at absolute prompt positions"))
      }
      if actual.work == expected.work,
        actual.work.keyValueProjectionInputTokens == model.layerCount * request.tokenIDs.count,
        actual.work.priorKeyValueTokensReprojected == 0
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "timing-independent prefill work model",
          message: "projection FLOPs, attention FLOPs, bytes, or projected-token counts differ"))
      }

      passed += expectError(name: "reject token outside vocabulary", failures: &failures) {
        let invalid = PromptPrefillRequest(model: model, tokenIDs: [model.vocabularySize])
        _ = try implementation(
          invalid,
          ContiguousKVCache(configuration: try model.cacheConfiguration(capacity: 1)))
      }
      passed += expectError(name: "reject nonempty prefill cache", failures: &failures) {
        let used = ContiguousKVCache(configuration: try model.cacheConfiguration(capacity: 4))
        let shape = [model.configuration.keyValueHeadCount, model.configuration.headDimension]
        let zeros = try FloatTensor(
          Array(repeating: 0, count: model.configuration.keyValueProjectionDimension),
          shape: shape)
        try used.append(layer: 0, logicalPosition: 0, key: zeros, value: zeros)
        _ = try implementation(
          PromptPrefillRequest(model: model, tokenIDs: [1]), used)
      }
      passed += expectError(name: "reject incompatible cache", failures: &failures) {
        let incompatible = ContiguousKVCache(configuration: try KVCacheConfiguration(
          layerCount: 1,
          keyValueHeadCount: model.configuration.keyValueHeadCount,
          headDimension: model.configuration.headDimension,
          capacity: 4))
        _ = try implementation(
          PromptPrefillRequest(model: model, tokenIDs: [1]), incompatible)
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 6, failures: failures)
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(caseName: name, message: "expected an error"))
      return 0
    } catch {
      return 1
    }
  }
}