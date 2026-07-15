import Foundation

public struct AutoregressiveDecodeRequest: Sendable, Equatable {
  public let model: MiniDecoderModel
  public let tokenID: Int
  public let logicalPosition: Int
  public let samplingStrategy: SamplingStrategy

  public init(
    model: MiniDecoderModel,
    tokenID: Int,
    logicalPosition: Int,
    samplingStrategy: SamplingStrategy
  ) {
    self.model = model
    self.tokenID = tokenID
    self.logicalPosition = logicalPosition
    self.samplingStrategy = samplingStrategy
  }
}

public struct AutoregressiveDecodeResult: Sendable, Equatable {
  public let inputTokenID: Int
  public let logicalPosition: Int
  public let selectedNextTokenID: Int
  public let sampling: SamplingTrace
  public let finalResidual: FloatTensor
  public let finalHidden: FloatTensor
  public let logits: FloatTensor
  public let layers: [MiniDecoderLayerTrace]
  public let cacheCountsBefore: [Int]
  public let cacheCountsAfter: [Int]
  public let cachePositions: [[Int]]
  public let work: MiniDecoderWorkModel

  public init(
    inputTokenID: Int,
    logicalPosition: Int,
    selectedNextTokenID: Int,
    sampling: SamplingTrace,
    finalResidual: FloatTensor,
    finalHidden: FloatTensor,
    logits: FloatTensor,
    layers: [MiniDecoderLayerTrace],
    cacheCountsBefore: [Int],
    cacheCountsAfter: [Int],
    cachePositions: [[Int]],
    work: MiniDecoderWorkModel
  ) {
    self.inputTokenID = inputTokenID
    self.logicalPosition = logicalPosition
    self.selectedNextTokenID = selectedNextTokenID
    self.sampling = sampling
    self.finalResidual = finalResidual
    self.finalHidden = finalHidden
    self.logits = logits
    self.layers = layers
    self.cacheCountsBefore = cacheCountsBefore
    self.cacheCountsAfter = cacheCountsAfter
    self.cachePositions = cachePositions
    self.work = work
  }
}

public typealias AutoregressiveDecodeImplementation = (
  _ request: AutoregressiveDecodeRequest,
  _ cache: ContiguousKVCache,
  _ generator: inout SeededGenerator
) throws -> AutoregressiveDecodeResult

public enum GenerationStopReason: String, Sendable, Equatable, Codable {
  case endOfSequence
  case maximumTokenCount
}

public struct MiniDecoderGenerationResult: Sendable, Equatable {
  public let promptTokenIDs: [Int]
  public let generatedTokenIDs: [Int]
  public let stopReason: GenerationStopReason
  public let prefill: PromptPrefillResult
  public let decodeSteps: [AutoregressiveDecodeResult]

  public init(
    promptTokenIDs: [Int],
    generatedTokenIDs: [Int],
    stopReason: GenerationStopReason,
    prefill: PromptPrefillResult,
    decodeSteps: [AutoregressiveDecodeResult]
  ) {
    self.promptTokenIDs = promptTokenIDs
    self.generatedTokenIDs = generatedTokenIDs
    self.stopReason = stopReason
    self.prefill = prefill
    self.decodeSteps = decodeSteps
  }
}

public enum P040AutoregressiveDecodeContract {
  public static func validate(
    request: AutoregressiveDecodeRequest,
    cache: any KVCacheReadable
  ) throws {
    try request.model.validate(tokenIDs: [request.tokenID])
    try request.model.validate(cache: cache)
    guard request.logicalPosition >= 0 else {
      throw KVCacheError.invalidLogicalPosition(request.logicalPosition)
    }
    try P038LogitsSamplingContract.validate(
      logits: Array(repeating: 0, count: request.model.vocabularySize),
      strategy: request.samplingStrategy)
    var expectedPositions: [Int]?
    for layer in 0..<request.model.layerCount {
      let positions = try cache.logicalPositions(layer: layer)
      if let expectedPositions, positions != expectedPositions {
        throw MiniDecoderError.cacheConfigurationMismatch
      }
      expectedPositions = positions
      if let last = positions.last {
        let (expected, overflow) = last.addingReportingOverflow(1)
        guard !overflow else { throw DecoderBlockError.positionOverflow }
        guard request.logicalPosition == expected else {
          throw MiniDecoderError.cachePositionMismatch(
            layer: layer, expected: expected, actual: request.logicalPosition)
        }
      }
      guard positions.count < cache.configuration.capacity else {
        throw KVCacheError.capacityExceeded(
          layer: layer, capacity: cache.configuration.capacity)
      }
    }
  }
}

public enum P040AutoregressiveDecodeJudge {
  public static func evaluate(_ implementation: AutoregressiveDecodeImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      let model = try EducationalMiniModelFixture.make()
      let positionOffset = 3
      var tokens = [1, 4]
      let prefix = try MiniDecoderReference.prefill(PromptPrefillRequest(
        model: model, tokenIDs: tokens, positionOffset: positionOffset))
      let cache = ContiguousKVCache(configuration: try model.cacheConfiguration(capacity: 8))
      try populate(cache: cache, from: prefix, model: model)
      var actualGenerator = SeededGenerator(seed: 0x040)
      var expectedGenerator = SeededGenerator(seed: 0x040)
      let strategy = SamplingStrategy.stochastic(
        SamplingConfiguration(temperature: 0.8, topK: 5, topP: 0.9))
      var allGrowthChecksPass = true

      for tokenID in [2, 3, 5] {
        let logicalPosition = positionOffset + tokens.count
        let before = try snapshots(cache: cache, model: model)
        let actual = try implementation(
          AutoregressiveDecodeRequest(
            model: model,
            tokenID: tokenID,
            logicalPosition: logicalPosition,
            samplingStrategy: strategy),
          cache,
          &actualGenerator)
        tokens.append(tokenID)
        let full = try MiniDecoderReference.prefill(PromptPrefillRequest(
          model: model, tokenIDs: tokens, positionOffset: positionOffset))
        let expectedSampling = try referenceSample(
          logits: full.logits.storage,
          strategy: strategy,
          generator: &expectedGenerator)
        let capturesMatch = try decodeCapturesMatch(actual, full: full)
        let cacheMatches = try cacheMatchesFull(cache, full: full, model: model)
        if MiniDecoderReference.approximatelyEqual(actual.finalHidden, full.finalHidden),
          MiniDecoderReference.approximatelyEqual(actual.logits, full.logits),
          samplingApproximatelyEqual(actual.sampling, expectedSampling),
          actual.selectedNextTokenID == expectedSampling.selectedToken,
          capturesMatch,
          cacheMatches
        {
          passed += 1
        } else {
          failures.append(JudgeFailure(
            caseName: "cached decode matches full recomputation at position \(logicalPosition)",
            message: "one-token captures, logits, sampling, or cache differ from the independent full-sequence oracle"))
        }
        let after = try snapshots(cache: cache, model: model)
        allGrowthChecksPass = allGrowthChecksPass
          && zip(actual.cacheCountsBefore, actual.cacheCountsAfter).allSatisfy { $1 == $0 + 1 }
          && actual.work.keyValueProjectionInputTokens == model.layerCount
          && actual.work.priorKeyValueTokensReprojected == 0
          && priorCachePrefixPreserved(before: before, after: after)
      }
      if allGrowthChecksPass, actualGenerator == expectedGenerator {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "one append and one KV projection per layer per token",
          message: "decode must preserve prior cache bytes, grow each layer once, and advance the sampler seed exactly once"))
      }

      passed += expectError(name: "reject cache position gap", failures: &failures) {
        let invalidCache = ContiguousKVCache(
          configuration: try model.cacheConfiguration(capacity: 4))
        try populate(cache: invalidCache, from: prefix, model: model)
        var generator = SeededGenerator(seed: 1)
        _ = try implementation(
          AutoregressiveDecodeRequest(
            model: model,
            tokenID: 2,
            logicalPosition: positionOffset + 4,
            samplingStrategy: .greedy),
          invalidCache,
          &generator)
      }
      passed += expectError(name: "reject invalid token", failures: &failures) {
        let invalidCache = ContiguousKVCache(
          configuration: try model.cacheConfiguration(capacity: 4))
        var generator = SeededGenerator(seed: 1)
        _ = try implementation(
          AutoregressiveDecodeRequest(
            model: model,
            tokenID: -1,
            logicalPosition: 0,
            samplingStrategy: .greedy),
          invalidCache,
          &generator)
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 6, failures: failures)
  }

  private static func populate(
    cache: ContiguousKVCache,
    from prefill: PromptPrefillResult,
    model: MiniDecoderModel
  ) throws {
    for layer in 0..<model.layerCount {
      let block = prefill.layers[layer].block.intermediates
      for token in 0..<prefill.promptTokenCount {
        try cache.append(
          layer: layer,
          logicalPosition: prefill.cachePositions[layer][token],
          key: try MiniDecoderReference.tokenSlice(
            block.rotatedKeys,
            token: token,
            headCount: model.configuration.keyValueHeadCount,
            headDimension: model.configuration.headDimension),
          value: try MiniDecoderReference.tokenSlice(
            block.values,
            token: token,
            headCount: model.configuration.keyValueHeadCount,
            headDimension: model.configuration.headDimension))
      }
    }
  }

  private static func decodeCapturesMatch(
    _ decode: AutoregressiveDecodeResult,
    full: PromptPrefillResult
  ) throws -> Bool {
    guard decode.layers.count == full.layers.count else { return false }
    for layer in full.layers.indices {
      let actual = decode.layers[layer]
      let expected = full.layers[layer]
      guard actual.layerIndex == expected.layerIndex,
        MiniDecoderReference.approximatelyEqual(
          actual.residualInput, try lastRow(expected.residualInput)),
        MiniDecoderReference.approximatelyEqual(
          actual.block.state.residual, try lastRow(expected.block.state.residual))
      else { return false }
      let left = actual.block.intermediates
      let right = expected.block.intermediates
      let pairs = [
        (left.attentionNormalized, right.attentionNormalized),
        (left.queries, right.queries),
        (left.keys, right.keys),
        (left.values, right.values),
        (left.rotatedQueries, right.rotatedQueries),
        (left.rotatedKeys, right.rotatedKeys),
        (left.attentionHeads, right.attentionHeads),
        (left.concatenatedAttention, right.concatenatedAttention),
        (left.attentionProjection, right.attentionProjection),
        (left.postAttentionResidual, right.postAttentionResidual),
        (left.mlpNormalized, right.mlpNormalized),
        (left.gateProjection, right.gateProjection),
        (left.upProjection, right.upProjection),
        (left.activatedGate, right.activatedGate),
        (left.gatedHidden, right.gatedHidden),
        (left.downProjection, right.downProjection),
      ]
      for (actualTensor, fullTensor) in pairs {
        guard MiniDecoderReference.approximatelyEqual(
          actualTensor, try lastRow(fullTensor))
        else { return false }
      }
    }
    return true
  }

  private static func lastRow(_ tensor: FloatTensor) throws -> FloatTensor {
    guard tensor.rank > 1 else { return tensor }
    let rowElements = tensor.shape.dropFirst().reduce(1, *)
    let start = tensor.storage.count - rowElements
    return try FloatTensor(
      Array(tensor.storage[start...]),
      shape: [1] + Array(tensor.shape.dropFirst()))
  }

  private static func cacheMatchesFull(
    _ cache: ContiguousKVCache,
    full: PromptPrefillResult,
    model: MiniDecoderModel
  ) throws -> Bool {
    for layer in 0..<model.layerCount {
      let materialized = try cache.materialized(layer: layer)
      guard materialized.positions == full.cachePositions[layer],
        MiniDecoderReference.approximatelyEqual(
          materialized.keys, full.layers[layer].block.intermediates.rotatedKeys),
        MiniDecoderReference.approximatelyEqual(
          materialized.values, full.layers[layer].block.intermediates.values)
      else { return false }
    }
    return true
  }

  private static func snapshots(
    cache: ContiguousKVCache,
    model: MiniDecoderModel
  ) throws -> [KVCacheLayerSnapshot] {
    try (0..<model.layerCount).map { layer in
      let materialized = try cache.materialized(layer: layer)
      return KVCacheLayerSnapshot(
        logicalPositions: materialized.positions,
        keys: materialized.keys,
        values: materialized.values)
    }
  }

  private static func priorCachePrefixPreserved(
    before: [KVCacheLayerSnapshot],
    after: [KVCacheLayerSnapshot]
  ) -> Bool {
    zip(before, after).allSatisfy { old, new in
      Array(new.logicalPositions.prefix(old.logicalPositions.count)) == old.logicalPositions
        && Array(new.keys.storage.prefix(old.keys.elementCount)) == old.keys.storage
        && Array(new.values.storage.prefix(old.values.elementCount)) == old.values.storage
    }
  }

  private struct RankedLogit {
    let tokenID: Int
    let logit: Float
  }

  private static func samplingApproximatelyEqual(
    _ lhs: SamplingTrace,
    _ rhs: SamplingTrace
  ) -> Bool {
    lhs.selectedToken == rhs.selectedToken
      && lhs.randomDraw == rhs.randomDraw
      && lhs.retainedCandidates.count == rhs.retainedCandidates.count
      && zip(lhs.retainedCandidates, rhs.retainedCandidates).allSatisfy { actual, expected in
        actual.tokenID == expected.tokenID
          && abs(actual.logit - expected.logit)
            <= MiniDecoderReference.absoluteTolerance
              + MiniDecoderReference.relativeTolerance * abs(expected.logit)
          && abs(actual.probability - expected.probability) <= 3e-5
      }
  }

  private static func referenceSample(
    logits: [Float],
    strategy: SamplingStrategy,
    generator: inout SeededGenerator
  ) throws -> SamplingTrace {
    try P038LogitsSamplingContract.validate(logits: logits, strategy: strategy)
    var ranked = logits.indices.map { RankedLogit(tokenID: $0, logit: logits[$0]) }
    ranked.sort { $0.logit == $1.logit ? $0.tokenID < $1.tokenID : $0.logit > $1.logit }
    guard case .stochastic(let configuration) = strategy else {
      return SamplingTrace(
        selectedToken: ranked[0].tokenID,
        retainedCandidates: [SamplingCandidate(
          tokenID: ranked[0].tokenID, logit: ranked[0].logit, probability: 1)],
        randomDraw: nil)
    }
    if let topK = configuration.topK { ranked = Array(ranked.prefix(topK)) }
    let scaled = ranked.map { Double($0.logit) / Double(configuration.temperature) }
    let maximum = scaled.max()!
    var probabilities = scaled.map { exp($0 - maximum) }
    let total = probabilities.reduce(0, +)
    probabilities = probabilities.map { $0 / total }
    if let topP = configuration.topP {
      var cumulative = 0.0
      var retained = 0
      repeat {
        cumulative += probabilities[retained]
        retained += 1
      } while retained < probabilities.count && cumulative < Double(topP)
      ranked = Array(ranked.prefix(retained))
      probabilities = Array(probabilities.prefix(retained))
    }
    let retainedTotal = probabilities.reduce(0, +)
    probabilities = probabilities.map { $0 / retainedTotal }
    let draw = generator.nextUnitInterval()
    var cumulative = 0.0
    var selected = ranked.last!.tokenID
    for index in ranked.indices {
      cumulative += probabilities[index]
      if draw < cumulative {
        selected = ranked[index].tokenID
        break
      }
    }
    return SamplingTrace(
      selectedToken: selected,
      retainedCandidates: ranked.indices.map {
        SamplingCandidate(
          tokenID: ranked[$0].tokenID,
          logit: ranked[$0].logit,
          probability: Float(probabilities[$0]))
      },
      randomDraw: draw)
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