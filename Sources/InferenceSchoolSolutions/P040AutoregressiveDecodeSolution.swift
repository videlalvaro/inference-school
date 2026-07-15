import InferenceSchoolCore

public enum P040AutoregressiveDecodeSolution {
  public static func run(
    _ request: AutoregressiveDecodeRequest,
    cache: ContiguousKVCache,
    generator: inout SeededGenerator
  ) throws -> AutoregressiveDecodeResult {
    try MiniDecoderCPUEngine.decode(request, cache: cache, generator: &generator)
  }
}

public final class MiniDecoderGenerationSession {
  public let model: MiniDecoderModel
  public let cache: ContiguousKVCache
  public let samplingStrategy: SamplingStrategy
  public let endOfSequenceTokenID: Int?
  public private(set) var generator: SeededGenerator

  private var nextLogicalPosition: Int?

  public init(
    model: MiniDecoderModel,
    cacheCapacity: Int,
    samplingStrategy: SamplingStrategy,
    seed: UInt64,
    endOfSequenceTokenID: Int? = nil
  ) throws {
    if let endOfSequenceTokenID,
      !(0..<model.vocabularySize).contains(endOfSequenceTokenID)
    {
      throw MiniDecoderError.invalidEOS(endOfSequenceTokenID)
    }
    self.model = model
    cache = ContiguousKVCache(
      configuration: try model.cacheConfiguration(capacity: cacheCapacity))
    self.samplingStrategy = samplingStrategy
    self.endOfSequenceTokenID = endOfSequenceTokenID
    generator = SeededGenerator(seed: seed)
  }

  public func prefill(
    tokenIDs: [Int],
    positionOffset: Int = 0
  ) throws -> PromptPrefillResult {
    let result = try MiniDecoderCPUEngine.prefill(
      PromptPrefillRequest(
        model: model, tokenIDs: tokenIDs, positionOffset: positionOffset),
      cache: cache)
    let (nextPosition, overflow) = positionOffset.addingReportingOverflow(tokenIDs.count)
    guard !overflow else { throw DecoderBlockError.positionOverflow }
    nextLogicalPosition = nextPosition
    return result
  }

  public func decode(tokenID: Int) throws -> AutoregressiveDecodeResult {
    guard let logicalPosition = nextLogicalPosition else {
      throw MiniDecoderError.emptyPrompt
    }
    let result = try MiniDecoderCPUEngine.decode(
      AutoregressiveDecodeRequest(
        model: model,
        tokenID: tokenID,
        logicalPosition: logicalPosition,
        samplingStrategy: samplingStrategy),
      cache: cache,
      generator: &generator)
    let (nextPosition, overflow) = logicalPosition.addingReportingOverflow(1)
    guard !overflow else { throw DecoderBlockError.positionOverflow }
    nextLogicalPosition = nextPosition
    return result
  }

  public func generate(
    promptTokenIDs: [Int],
    maxNewTokens: Int,
    positionOffset: Int = 0
  ) throws -> MiniDecoderGenerationResult {
    guard maxNewTokens >= 0 else {
      throw MiniDecoderError.invalidGenerationLimit(maxNewTokens)
    }
    let prefillResult = try prefill(
      tokenIDs: promptTokenIDs, positionOffset: positionOffset)
    guard maxNewTokens > 0 else {
      return MiniDecoderGenerationResult(
        promptTokenIDs: promptTokenIDs,
        generatedTokenIDs: [],
        stopReason: .maximumTokenCount,
        prefill: prefillResult,
        decodeSteps: [])
    }
    let firstSampling = try P038LogitsSamplingSolution.sample(
      logits: prefillResult.logits.storage,
      strategy: samplingStrategy,
      generator: &generator)
    var generated = [firstSampling.selectedToken]
    if generated.last == endOfSequenceTokenID {
      return MiniDecoderGenerationResult(
        promptTokenIDs: promptTokenIDs,
        generatedTokenIDs: generated,
        stopReason: .endOfSequence,
        prefill: prefillResult,
        decodeSteps: [])
    }
    var steps: [AutoregressiveDecodeResult] = []
    while generated.count < maxNewTokens {
      let step = try decode(tokenID: generated.last!)
      steps.append(step)
      generated.append(step.selectedNextTokenID)
      if generated.last == endOfSequenceTokenID {
        return MiniDecoderGenerationResult(
          promptTokenIDs: promptTokenIDs,
          generatedTokenIDs: generated,
          stopReason: .endOfSequence,
          prefill: prefillResult,
          decodeSteps: steps)
      }
    }
    return MiniDecoderGenerationResult(
      promptTokenIDs: promptTokenIDs,
      generatedTokenIDs: generated,
      stopReason: .maximumTokenCount,
      prefill: prefillResult,
      decodeSteps: steps)
  }
}