import Foundation

public enum MiniDecoderError: Error, Equatable, LocalizedError {
  case invalidVocabularySize(Int)
  case emptyLayerStack
  case emptyPrompt
  case tokenOutOfVocabulary(index: Int, tokenID: Int, vocabularySize: Int)
  case invalidCacheCapacity(Int)
  case cacheConfigurationMismatch
  case cacheNotEmpty(layer: Int, count: Int)
  case cachePositionMismatch(layer: Int, expected: Int, actual: Int)
  case invalidGenerationLimit(Int)
  case invalidEOS(Int)
  case integerOverflow(context: String)

  public var errorDescription: String? {
    switch self {
    case .invalidVocabularySize(let value):
      "Vocabulary size must be positive; received \(value)."
    case .emptyLayerStack:
      "The educational mini-model requires at least one decoder block."
    case .emptyPrompt:
      "Prompt prefill requires at least one token ID."
    case .tokenOutOfVocabulary(let index, let tokenID, let vocabularySize):
      "Token ID \(tokenID) at index \(index) is outside 0..<\(vocabularySize)."
    case .invalidCacheCapacity(let value):
      "Cache capacity must be positive; received \(value)."
    case .cacheConfigurationMismatch:
      "The KV cache does not match the model layer, KV-head, or head-dimension contract."
    case .cacheNotEmpty(let layer, let count):
      "Prompt prefill requires an empty cache; layer \(layer) already contains \(count) tokens."
    case .cachePositionMismatch(let layer, let expected, let actual):
      "Layer \(layer) expected cache position \(expected); received \(actual)."
    case .invalidGenerationLimit(let value):
      "Maximum generated token count must be nonnegative; received \(value)."
    case .invalidEOS(let value):
      "EOS token ID \(value) is outside the model vocabulary."
    case .integerOverflow(let context):
      "Integer arithmetic overflowed while computing \(context)."
    }
  }
}

public enum MiniDecoderOutputProjection: Sendable, Equatable {
  case tiedEmbedding
  case independent(FloatTensor)
}

public struct MiniDecoderModel: Sendable, Equatable {
  public let vocabularySize: Int
  public let configuration: DecoderConfiguration
  public let tokenEmbedding: FloatTensor
  public let blocks: [DecoderBlockWeights]
  public let finalNormGamma: FloatTensor
  public let outputProjection: MiniDecoderOutputProjection

  public var layerCount: Int { blocks.count }
  public var outputWeights: FloatTensor {
    switch outputProjection {
    case .tiedEmbedding:
      tokenEmbedding
    case .independent(let weights):
      weights
    }
  }

  public init(
    vocabularySize: Int,
    configuration: DecoderConfiguration,
    tokenEmbedding: FloatTensor,
    blocks: [DecoderBlockWeights],
    finalNormGamma: FloatTensor,
    outputProjection: MiniDecoderOutputProjection
  ) throws {
    guard vocabularySize > 0 else {
      throw MiniDecoderError.invalidVocabularySize(vocabularySize)
    }
    guard !blocks.isEmpty else { throw MiniDecoderError.emptyLayerStack }
    let modelDimension = configuration.modelDimension
    try Self.validate(
      tokenEmbedding,
      name: "Token embedding",
      expectedShape: [vocabularySize, modelDimension])
    for (index, weights) in blocks.enumerated() {
      do {
        try P035DecoderBlockContract.validateWeights(weights, configuration: configuration)
      } catch let error as DecoderBlockError {
        switch error {
        case .shapeMismatch(let tensor, let expected, let actual):
          throw DecoderBlockError.shapeMismatch(
            tensor: "Layer \(index) \(tensor)", expected: expected, actual: actual)
        case .nonFiniteValue(let tensor, let linearIndex):
          throw DecoderBlockError.nonFiniteValue(
            tensor: "Layer \(index) \(tensor)", linearIndex: linearIndex)
        default:
          throw error
        }
      }
    }
    try Self.validate(
      finalNormGamma,
      name: "Final RMSNorm gamma",
      expectedShape: [modelDimension])
    if case .independent(let weights) = outputProjection {
      try Self.validate(
        weights,
        name: "Output weights",
        expectedShape: [vocabularySize, modelDimension])
    }
    self.vocabularySize = vocabularySize
    self.configuration = configuration
    self.tokenEmbedding = tokenEmbedding
    self.blocks = blocks
    self.finalNormGamma = finalNormGamma
    self.outputProjection = outputProjection
  }

  public func cacheConfiguration(capacity: Int) throws -> KVCacheConfiguration {
    guard capacity > 0 else { throw MiniDecoderError.invalidCacheCapacity(capacity) }
    return try KVCacheConfiguration(
      layerCount: layerCount,
      keyValueHeadCount: configuration.keyValueHeadCount,
      headDimension: configuration.headDimension,
      capacity: capacity)
  }

  public func validate(cache: any KVCacheReadable) throws {
    let cacheConfiguration = cache.configuration
    guard cacheConfiguration.layerCount == layerCount,
      cacheConfiguration.keyValueHeadCount == configuration.keyValueHeadCount,
      cacheConfiguration.headDimension == configuration.headDimension
    else {
      throw MiniDecoderError.cacheConfigurationMismatch
    }
  }

  public func validate(tokenIDs: [Int], allowEmpty: Bool = false) throws {
    if !allowEmpty, tokenIDs.isEmpty { throw MiniDecoderError.emptyPrompt }
    for (index, tokenID) in tokenIDs.enumerated()
      where tokenID < 0 || tokenID >= vocabularySize
    {
      throw MiniDecoderError.tokenOutOfVocabulary(
        index: index, tokenID: tokenID, vocabularySize: vocabularySize)
    }
  }

  public var fingerprint: String {
    var hash: UInt64 = 0xcbf29ce484222325
    func mixByte(_ byte: UInt8) {
      hash ^= UInt64(byte)
      hash &*= 0x100000001b3
    }
    func mixInteger(_ value: Int) {
      var bits = UInt64(truncatingIfNeeded: value)
      for _ in 0..<8 {
        mixByte(UInt8(truncatingIfNeeded: bits))
        bits >>= 8
      }
    }
    func mixFloat(_ value: Float) {
      var bits = value.bitPattern
      for _ in 0..<4 {
        mixByte(UInt8(truncatingIfNeeded: bits))
        bits >>= 8
      }
    }
    func mixTensor(_ tensor: FloatTensor) {
      mixInteger(tensor.shape.count)
      tensor.shape.forEach(mixInteger)
      tensor.storage.forEach(mixFloat)
    }

    mixInteger(vocabularySize)
    for value in [
      configuration.modelDimension,
      configuration.hiddenDimension,
      configuration.queryHeadCount,
      configuration.keyValueHeadCount,
      configuration.headDimension,
      configuration.rotaryDimension,
      layerCount,
    ] { mixInteger(value) }
    mixFloat(configuration.rmsNormEpsilon)
    mixFloat(configuration.ropeBase)
    mixTensor(tokenEmbedding)
    for block in blocks {
      mixTensor(block.attentionNormGamma)
      mixTensor(block.queryWeights)
      mixTensor(block.keyWeights)
      mixTensor(block.valueWeights)
      mixTensor(block.attentionOutputWeights)
      mixTensor(block.mlpNormGamma)
      mixTensor(block.gateWeights)
      mixTensor(block.upWeights)
      mixTensor(block.downWeights)
    }
    mixTensor(finalNormGamma)
    switch outputProjection {
    case .tiedEmbedding:
      mixByte(0)
    case .independent(let weights):
      mixByte(1)
      mixTensor(weights)
    }
    return String(format: "%016llx", hash)
  }

  private static func validate(
    _ tensor: FloatTensor,
    name: String,
    expectedShape: [Int]
  ) throws {
    guard tensor.shape == expectedShape else {
      throw DecoderBlockError.shapeMismatch(
        tensor: name, expected: expectedShape, actual: tensor.shape)
    }
    if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
      throw DecoderBlockError.nonFiniteValue(tensor: name, linearIndex: index)
    }
  }
}

public struct MiniDecoderWorkModel: Sendable, Equatable {
  public let projectionFLOPs: Int
  public let attentionFLOPs: Int
  public let estimatedWeightBytesRead: Int
  public let cacheBytesWritten: Int
  public let keyValueProjectionInputTokens: Int
  public let priorKeyValueTokensReprojected: Int

  public init(
    projectionFLOPs: Int,
    attentionFLOPs: Int,
    estimatedWeightBytesRead: Int,
    cacheBytesWritten: Int,
    keyValueProjectionInputTokens: Int,
    priorKeyValueTokensReprojected: Int
  ) {
    self.projectionFLOPs = projectionFLOPs
    self.attentionFLOPs = attentionFLOPs
    self.estimatedWeightBytesRead = estimatedWeightBytesRead
    self.cacheBytesWritten = cacheBytesWritten
    self.keyValueProjectionInputTokens = keyValueProjectionInputTokens
    self.priorKeyValueTokensReprojected = priorKeyValueTokensReprojected
  }
}

public struct MiniDecoderLayerTrace: Sendable, Equatable {
  public let layerIndex: Int
  public let residualInput: FloatTensor
  public let block: DecoderBlockResult
  public let cachePositions: [Int]

  public init(
    layerIndex: Int,
    residualInput: FloatTensor,
    block: DecoderBlockResult,
    cachePositions: [Int]
  ) {
    self.layerIndex = layerIndex
    self.residualInput = residualInput
    self.block = block
    self.cachePositions = cachePositions
  }
}

public enum EducationalMiniModelFixture {
  public static let defaultPrompt = [1, 4, 2]
  public static let endOfSequenceTokenID = 0

  public static func make(layerCount: Int = 2) throws -> MiniDecoderModel {
    guard layerCount > 0 else { throw MiniDecoderError.emptyLayerStack }
    let configuration = try DecoderConfiguration(
      modelDimension: 4,
      hiddenDimension: 6,
      queryHeadCount: 2,
      keyValueHeadCount: 1,
      headDimension: 2,
      rotaryDimension: 2,
      rmsNormEpsilon: 1e-5,
      ropeBase: 100)
    let vocabularySize = 7

    func tensor(_ shape: [Int], salt: Int, scale: Float = 17) throws -> FloatTensor {
      let count = shape.reduce(1, *)
      return try FloatTensor((0..<count).map { index in
        Float(((index * 11 + salt * 7) % 29) - 14) / scale
      }, shape: shape)
    }
    func block(_ layer: Int) throws -> DecoderBlockWeights {
      let salt = 20 * layer
      return DecoderBlockWeights(
        attentionNormGamma: try FloatTensor(
          (0..<4).map { 0.75 + Float((layer + $0) % 5) * 0.1 }, shape: [4]),
        queryWeights: try tensor([4, 4], salt: salt + 1),
        keyWeights: try tensor([2, 4], salt: salt + 2),
        valueWeights: try tensor([2, 4], salt: salt + 3),
        attentionOutputWeights: try tensor([4, 4], salt: salt + 4),
        mlpNormGamma: try FloatTensor(
          (0..<4).map { 0.8 + Float((layer * 2 + $0) % 4) * 0.12 }, shape: [4]),
        gateWeights: try tensor([6, 4], salt: salt + 5, scale: 23),
        upWeights: try tensor([6, 4], salt: salt + 6, scale: 23),
        downWeights: try tensor([4, 6], salt: salt + 7, scale: 23))
    }

    return try MiniDecoderModel(
      vocabularySize: vocabularySize,
      configuration: configuration,
      tokenEmbedding: tensor([vocabularySize, 4], salt: 91, scale: 19),
      blocks: try (0..<layerCount).map(block),
      finalNormGamma: FloatTensor([1.05, 0.9, 1.15, 0.8], shape: [4]),
      outputProjection: .tiedEmbedding)
  }
}