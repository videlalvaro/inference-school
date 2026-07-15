import InferenceSchoolCore

public enum P040AutoregressiveDecodeExercise {
  public static func run(
    _ request: AutoregressiveDecodeRequest,
    cache: ContiguousKVCache,
    generator: inout SeededGenerator
  ) throws -> AutoregressiveDecodeResult {
    try P040AutoregressiveDecodeContract.validate(request: request, cache: cache)
    let dimension = request.model.configuration.modelDimension
    let hidden = try FloatTensor(Array(repeating: 0, count: dimension), shape: [dimension])
    let logits = try FloatTensor(
      Array(repeating: 0, count: request.model.vocabularySize),
      shape: [request.model.vocabularySize])
    let sampling = SamplingTrace(
      selectedToken: 0,
      retainedCandidates: [SamplingCandidate(tokenID: 0, logit: 0, probability: 1)],
      randomDraw: nil)

    // TODO: project only the current token, rotate at logicalPosition, append one
    // K/V record per layer, attend over the existing cache, and sample the logits.
    return AutoregressiveDecodeResult(
      inputTokenID: request.tokenID,
      logicalPosition: request.logicalPosition,
      selectedNextTokenID: sampling.selectedToken,
      sampling: sampling,
      finalResidual: hidden,
      finalHidden: hidden,
      logits: logits,
      layers: [],
      cacheCountsBefore: try (0..<request.model.layerCount).map {
        try cache.count(layer: $0)
      },
      cacheCountsAfter: try (0..<request.model.layerCount).map {
        try cache.count(layer: $0)
      },
      cachePositions: try (0..<request.model.layerCount).map {
        try cache.logicalPositions(layer: $0)
      },
      work: MiniDecoderWorkModel(
        projectionFLOPs: 0,
        attentionFLOPs: 0,
        estimatedWeightBytesRead: 0,
        cacheBytesWritten: 0,
        keyValueProjectionInputTokens: 0,
        priorKeyValueTokensReprojected: 0))
  }
}