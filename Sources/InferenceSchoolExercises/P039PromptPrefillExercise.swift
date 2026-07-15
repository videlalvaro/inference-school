import InferenceSchoolCore

public enum P039PromptPrefillExercise {
  public static func run(
    _ request: PromptPrefillRequest,
    cache: ContiguousKVCache
  ) throws -> PromptPrefillResult {
    try P039PromptPrefillContract.validate(request: request, cache: cache)
    let sequence = request.tokenIDs.count
    let dimension = request.model.configuration.modelDimension
    let residual = try FloatTensor(
      Array(repeating: 0, count: sequence * dimension),
      shape: [sequence, dimension])
    let hidden = try FloatTensor(Array(repeating: 0, count: dimension), shape: [dimension])
    let logits = try FloatTensor(
      Array(repeating: 0, count: request.model.vocabularySize),
      shape: [request.model.vocabularySize])

    // TODO: embed all prompt tokens, run every block in order, append each layer's
    // rotated K and V to the cache, then normalize and unembed the final position.
    return PromptPrefillResult(
      promptTokenCount: sequence,
      finalResidual: residual,
      finalNormalized: residual,
      finalHidden: hidden,
      logits: logits,
      layers: [],
      cacheCounts: Array(repeating: 0, count: request.model.layerCount),
      cachePositions: Array(repeating: [], count: request.model.layerCount),
      work: MiniDecoderWorkModel(
        projectionFLOPs: 0,
        attentionFLOPs: 0,
        estimatedWeightBytesRead: 0,
        cacheBytesWritten: 0,
        keyValueProjectionInputTokens: 0,
        priorKeyValueTokensReprojected: 0))
  }
}