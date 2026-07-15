import InferenceSchoolCore

public enum P035DecoderBlockExercise {
  public static func apply(
    state: DecoderBlockState,
    weights: DecoderBlockWeights,
    configuration: DecoderConfiguration
  ) throws -> DecoderBlockResult {
    try P035DecoderBlockContract.validate(
      state: state, weights: weights, configuration: configuration)

    let sequence = state.residual.shape[0]
    let model = configuration.modelDimension
    let hidden = configuration.hiddenDimension
    let queryShape = [sequence, configuration.queryHeadCount, configuration.headDimension]
    let keyValueShape = [sequence, configuration.keyValueHeadCount, configuration.headDimension]
    let modelShape = [sequence, model]
    let hiddenShape = [sequence, hidden]
    let zeroModel = try FloatTensor(Array(repeating: 0, count: sequence * model), shape: modelShape)
    let zeroQuery = try FloatTensor(
      Array(repeating: 0, count: sequence * configuration.queryProjectionDimension),
      shape: queryShape)
    let zeroKeyValue = try FloatTensor(
      Array(repeating: 0, count: sequence * configuration.keyValueProjectionDimension),
      shape: keyValueShape)
    let zeroHidden = try FloatTensor(
      Array(repeating: 0, count: sequence * hidden), shape: hiddenShape)

    // TODO: implement the ordered pre-norm attention and SwiGLU block.
    return DecoderBlockResult(
      state: state,
      intermediates: DecoderBlockIntermediates(
        attentionNormalized: zeroModel,
        queries: zeroQuery,
        keys: zeroKeyValue,
        values: zeroKeyValue,
        rotatedQueries: zeroQuery,
        rotatedKeys: zeroKeyValue,
        attentionHeads: zeroQuery,
        concatenatedAttention: zeroModel,
        attentionProjection: zeroModel,
        postAttentionResidual: state.residual,
        mlpNormalized: zeroModel,
        gateProjection: zeroHidden,
        upProjection: zeroHidden,
        activatedGate: zeroHidden,
        gatedHidden: zeroHidden,
        downProjection: zeroModel))
  }
}