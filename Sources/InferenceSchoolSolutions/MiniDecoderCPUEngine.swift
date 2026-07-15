import Foundation
import InferenceSchoolCore

public enum MiniDecoderCPUEngine {
  public static func prefill(
    _ request: PromptPrefillRequest,
    cache: ContiguousKVCache
  ) throws -> PromptPrefillResult {
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
      let result = try P035DecoderBlockSolution.apply(
        state: DecoderBlockState(
          residual: input, positionOffset: request.positionOffset),
        weights: weights,
        configuration: model.configuration)
      for token in 0..<sequence {
        try cache.append(
          layer: layer,
          logicalPosition: request.positionOffset + token,
          key: try tokenSlice(
            result.intermediates.rotatedKeys,
            token: token,
            headCount: model.configuration.keyValueHeadCount,
            headDimension: model.configuration.headDimension),
          value: try tokenSlice(
            result.intermediates.values,
            token: token,
            headCount: model.configuration.keyValueHeadCount,
            headDimension: model.configuration.headDimension))
      }
      residual = result.state.residual
      traces.append(MiniDecoderLayerTrace(
        layerIndex: layer,
        residualInput: input,
        block: result,
        cachePositions: try cache.logicalPositions(layer: layer)))
    }
    let finalNormalized = try normalize(
      residual, gamma: model.finalNormGamma, epsilon: model.configuration.rmsNormEpsilon)
    let lastStart = (sequence - 1) * dimension
    let finalHidden = try FloatTensor(
      Array(finalNormalized.storage[lastStart..<(lastStart + dimension)]),
      shape: [dimension])
    let logits = try projectVector(finalHidden, weights: model.outputWeights)
    return PromptPrefillResult(
      promptTokenCount: sequence,
      finalResidual: residual,
      finalNormalized: finalNormalized,
      finalHidden: finalHidden,
      logits: logits,
      layers: traces,
      cacheCounts: try (0..<model.layerCount).map { try cache.count(layer: $0) },
      cachePositions: try (0..<model.layerCount).map {
        try cache.logicalPositions(layer: $0)
      },
      work: try workModel(model: model, tokenCount: sequence, decode: false))
  }

  public static func decode(
    _ request: AutoregressiveDecodeRequest,
    cache: ContiguousKVCache,
    generator: inout SeededGenerator
  ) throws -> AutoregressiveDecodeResult {
    try P040AutoregressiveDecodeContract.validate(request: request, cache: cache)
    let model = request.model
    let configuration = model.configuration
    let dimension = configuration.modelDimension
    let embeddingStart = request.tokenID * dimension
    var residual = try FloatTensor(
      Array(model.tokenEmbedding.storage[embeddingStart..<(embeddingStart + dimension)]),
      shape: [1, dimension])
    let countsBefore = try (0..<model.layerCount).map { try cache.count(layer: $0) }
    var traces: [MiniDecoderLayerTrace] = []

    for (layer, weights) in model.blocks.enumerated() {
      let residualInput = residual
      let attentionNormalized = try normalize(
        residual, gamma: weights.attentionNormGamma, epsilon: configuration.rmsNormEpsilon)
      let queryProjection = try project(attentionNormalized, weights: weights.queryWeights)
      let keyProjection = try project(attentionNormalized, weights: weights.keyWeights)
      let valueProjection = try project(attentionNormalized, weights: weights.valueWeights)
      let queries = try FloatTensor(
        queryProjection.storage,
        shape: [1, configuration.queryHeadCount, configuration.headDimension])
      let keys = try FloatTensor(
        keyProjection.storage,
        shape: [1, configuration.keyValueHeadCount, configuration.headDimension])
      let values = try FloatTensor(
        valueProjection.storage,
        shape: [1, configuration.keyValueHeadCount, configuration.headDimension])
      let rotatedQueries = try rotate(
        queries, positionOffset: request.logicalPosition, configuration: configuration)
      let rotatedKeys = try rotate(
        keys, positionOffset: request.logicalPosition, configuration: configuration)
      let queryVector = try tokenSlice(
        rotatedQueries,
        token: 0,
        headCount: configuration.queryHeadCount,
        headDimension: configuration.headDimension)
      try cache.append(
        layer: layer,
        logicalPosition: request.logicalPosition,
        key: try tokenSlice(
          rotatedKeys,
          token: 0,
          headCount: configuration.keyValueHeadCount,
          headDimension: configuration.headDimension),
        value: try tokenSlice(
          values,
          token: 0,
          headCount: configuration.keyValueHeadCount,
          headDimension: configuration.headDimension))
      let attended = try P023CachedAttentionSolution.attend(
        query: queryVector,
        cache: cache,
        layer: layer,
        queryLogicalPosition: request.logicalPosition,
        queryHeadCount: configuration.queryHeadCount)
      let attentionHeads = try FloatTensor(
        attended.storage,
        shape: [1, configuration.queryHeadCount, configuration.headDimension])
      let concatenatedAttention = try FloatTensor(attended.storage, shape: [1, dimension])
      let attentionProjection = try project(
        concatenatedAttention, weights: weights.attentionOutputWeights)
      let postAttentionResidual = try add(residual, attentionProjection)
      let mlpNormalized = try normalize(
        postAttentionResidual,
        gamma: weights.mlpNormGamma,
        epsilon: configuration.rmsNormEpsilon)
      let gateProjection = try project(mlpNormalized, weights: weights.gateWeights)
      let upProjection = try project(mlpNormalized, weights: weights.upWeights)
      let activatedGate = try FloatTensor(
        gateProjection.storage.map { $0 / (1 + exp(-$0)) },
        shape: gateProjection.shape)
      let gatedHidden = try FloatTensor(
        zip(activatedGate.storage, upProjection.storage).map(*),
        shape: gateProjection.shape)
      let downProjection = try project(gatedHidden, weights: weights.downWeights)
      residual = try add(postAttentionResidual, downProjection)
      let result = DecoderBlockResult(
        state: DecoderBlockState(
          residual: residual, positionOffset: request.logicalPosition),
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
      traces.append(MiniDecoderLayerTrace(
        layerIndex: layer,
        residualInput: residualInput,
        block: result,
        cachePositions: try cache.logicalPositions(layer: layer)))
    }
    let finalHiddenMatrix = try normalize(
      residual, gamma: model.finalNormGamma, epsilon: configuration.rmsNormEpsilon)
    let finalHidden = try FloatTensor(finalHiddenMatrix.storage, shape: [dimension])
    let logits = try projectVector(finalHidden, weights: model.outputWeights)
    let sampling = try P038LogitsSamplingSolution.sample(
      logits: logits.storage,
      strategy: request.samplingStrategy,
      generator: &generator)
    let countsAfter = try (0..<model.layerCount).map { try cache.count(layer: $0) }
    let visibleTokenCount = countsAfter[0]
    return AutoregressiveDecodeResult(
      inputTokenID: request.tokenID,
      logicalPosition: request.logicalPosition,
      selectedNextTokenID: sampling.selectedToken,
      sampling: sampling,
      finalResidual: try FloatTensor(residual.storage, shape: [dimension]),
      finalHidden: finalHidden,
      logits: logits,
      layers: traces,
      cacheCountsBefore: countsBefore,
      cacheCountsAfter: countsAfter,
      cachePositions: try (0..<model.layerCount).map {
        try cache.logicalPositions(layer: $0)
      },
      work: try workModel(
        model: model,
        tokenCount: 1,
        attentionTokenCount: visibleTokenCount,
        decode: true))
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
      var sumSquares: Float = 0
      for column in 0..<width {
        let value = input.storage[row * width + column]
        sumSquares += value * value
      }
      let inverseRMS = 1 / sqrt(sumSquares / Float(width) + epsilon)
      for column in 0..<width {
        let scale = gamma.storage[column] + (additiveGamma ? 1 : 0)
        output[row * width + column] = input.storage[row * width + column] * inverseRMS * scale
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
        var sum: Float = 0
        for inputChannel in 0..<inputWidth {
          sum += input.storage[row * inputWidth + inputChannel]
            * weights.storage[outputChannel * inputWidth + inputChannel]
        }
        output[row * outputWidth + outputChannel] = sum
      }
    }
    return try FloatTensor(output, shape: [rows, outputWidth])
  }

  static func projectVector(_ input: FloatTensor, weights: FloatTensor) throws -> FloatTensor {
    let matrix = try FloatTensor(input.storage, shape: [1, input.shape[0]])
    let output = try project(matrix, weights: weights)
    return try FloatTensor(output.storage, shape: [weights.shape[0]])
  }

  static func rotate(
    _ input: FloatTensor,
    positionOffset: Int,
    configuration: DecoderConfiguration
  ) throws -> FloatTensor {
    var output = input.storage
    for token in 0..<input.shape[0] {
      let position = Float(positionOffset + token)
      for head in 0..<input.shape[1] {
        let start = (token * input.shape[1] + head) * configuration.headDimension
        for pairStart in stride(from: 0, to: configuration.rotaryDimension, by: 2) {
          let pair = pairStart / 2
          let angle = position / pow(
            configuration.ropeBase,
            Float(2 * pair) / Float(configuration.rotaryDimension))
          let first = input.storage[start + pairStart]
          let second = input.storage[start + pairStart + 1]
          output[start + pairStart] = first * cos(angle) - second * sin(angle)
          output[start + pairStart + 1] = first * sin(angle) + second * cos(angle)
        }
      }
    }
    return try FloatTensor(output, shape: input.shape)
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
    let layerTerms = dimension * dimension + 2 * dimension * keyValueWidth
      + dimension * dimension + 3 * dimension * hidden
    let projectionFLOPs = 2 * tokenCount * model.layerCount * layerTerms
      + 2 * model.vocabularySize * dimension
    let visiblePairs = decode
      ? (attentionTokenCount ?? tokenCount)
      : tokenCount * (tokenCount + 1) / 2
    let attentionFLOPs = 4 * visiblePairs * configuration.queryHeadCount
      * configuration.headDimension * model.layerCount
    let weightElements = model.blocks.reduce(model.outputWeights.elementCount) { partial, block in
      partial + block.queryWeights.elementCount + block.keyWeights.elementCount
        + block.valueWeights.elementCount + block.attentionOutputWeights.elementCount
        + block.gateWeights.elementCount + block.upWeights.elementCount
        + block.downWeights.elementCount + block.attentionNormGamma.elementCount
        + block.mlpNormGamma.elementCount
    } + model.finalNormGamma.elementCount
    return MiniDecoderWorkModel(
      projectionFLOPs: projectionFLOPs,
      attentionFLOPs: attentionFLOPs,
      estimatedWeightBytesRead: weightElements * MemoryLayout<Float>.stride,
      cacheBytesWritten: tokenCount * model.layerCount * 2
        * configuration.keyValueHeadCount * configuration.headDimension
        * MemoryLayout<Float>.stride,
      keyValueProjectionInputTokens: tokenCount * model.layerCount,
      priorKeyValueTokensReprojected: 0)
  }

  static func add(_ lhs: FloatTensor, _ rhs: FloatTensor) throws -> FloatTensor {
    try FloatTensor(zip(lhs.storage, rhs.storage).map(+), shape: lhs.shape)
  }
}