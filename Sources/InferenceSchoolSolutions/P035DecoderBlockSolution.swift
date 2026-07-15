import Foundation
import InferenceSchoolCore

public enum P035DecoderBlockSolution {
  public static func apply(
    state: DecoderBlockState,
    weights: DecoderBlockWeights,
    configuration: DecoderConfiguration
  ) throws -> DecoderBlockResult {
    try P035DecoderBlockContract.validate(
      state: state, weights: weights, configuration: configuration)
    let sequenceLength = state.residual.shape[0]

    func normalize(_ input: FloatTensor, gamma: FloatTensor) throws -> FloatTensor {
      let width = input.shape[1]
      var output = Array(repeating: Float.zero, count: input.elementCount)
      for row in 0..<input.shape[0] {
        var sumSquares: Float = 0
        for column in 0..<width {
          let value = input.storage[row * width + column]
          sumSquares += value * value
        }
        let inverseRMS = 1 / sqrt(sumSquares / Float(width) + configuration.rmsNormEpsilon)
        for column in 0..<width {
          output[row * width + column] =
            input.storage[row * width + column] * inverseRMS * gamma.storage[column]
        }
      }
      return try FloatTensor(output, shape: input.shape)
    }

    func project(_ input: FloatTensor, weights: FloatTensor) throws -> FloatTensor {
      let inputWidth = input.shape[1]
      let outputWidth = weights.shape[0]
      var output = Array(repeating: Float.zero, count: input.shape[0] * outputWidth)
      for row in 0..<input.shape[0] {
        for outputChannel in 0..<outputWidth {
          var sum: Float = 0
          for inputChannel in 0..<inputWidth {
            sum += input.storage[row * inputWidth + inputChannel]
              * weights.storage[outputChannel * inputWidth + inputChannel]
          }
          output[row * outputWidth + outputChannel] = sum
        }
      }
      return try FloatTensor(output, shape: [input.shape[0], outputWidth])
    }

    func reshapeHeads(_ projection: FloatTensor, headCount: Int) throws -> FloatTensor {
      try FloatTensor(
        projection.storage,
        shape: [sequenceLength, headCount, configuration.headDimension])
    }

    func rotate(_ input: FloatTensor) throws -> FloatTensor {
      var output = input.storage
      for token in 0..<sequenceLength {
        let position = Float(state.positionOffset + token)
        for head in 0..<input.shape[1] {
          let start = (token * input.shape[1] + head) * configuration.headDimension
          for pairStart in stride(from: 0, to: configuration.rotaryDimension, by: 2) {
            let pair = pairStart / 2
            let angle = position / pow(
              configuration.ropeBase,
              Float(2 * pair) / Float(configuration.rotaryDimension))
            let cosine = cos(angle)
            let sine = sin(angle)
            let first = input.storage[start + pairStart]
            let second = input.storage[start + pairStart + 1]
            output[start + pairStart] = first * cosine - second * sine
            output[start + pairStart + 1] = first * sine + second * cosine
          }
        }
      }
      return try FloatTensor(output, shape: input.shape)
    }

    func causalAttention(
      queries: FloatTensor,
      keys: FloatTensor,
      values: FloatTensor
    ) throws -> FloatTensor {
      let headDimension = configuration.headDimension
      let groupSize = configuration.queryHeadCount / configuration.keyValueHeadCount
      let scale = 1 / sqrt(Float(headDimension))
      var output = Array(repeating: Float.zero, count: queries.elementCount)
      for query in 0..<sequenceLength {
        for queryHead in 0..<configuration.queryHeadCount {
          let keyValueHead = queryHead / groupSize
          var scores = Array(repeating: Float.zero, count: query + 1)
          for key in 0...query {
            var dot: Float = 0
            for feature in 0..<headDimension {
              let queryIndex =
                (query * configuration.queryHeadCount + queryHead) * headDimension + feature
              let keyIndex =
                (key * configuration.keyValueHeadCount + keyValueHead) * headDimension + feature
              dot += queries.storage[queryIndex] * keys.storage[keyIndex]
            }
            scores[key] = dot * scale
          }
          let maximum = scores.max()!
          let exponentials = scores.map { exp($0 - maximum) }
          let denominator = exponentials.reduce(0, +)
          for feature in 0..<headDimension {
            var sum: Float = 0
            for key in 0...query {
              let valueIndex =
                (key * configuration.keyValueHeadCount + keyValueHead) * headDimension + feature
              sum += exponentials[key] / denominator * values.storage[valueIndex]
            }
            let outputIndex =
              (query * configuration.queryHeadCount + queryHead) * headDimension + feature
            output[outputIndex] = sum
          }
        }
      }
      return try FloatTensor(output, shape: queries.shape)
    }

    func add(_ lhs: FloatTensor, _ rhs: FloatTensor) throws -> FloatTensor {
      try FloatTensor(zip(lhs.storage, rhs.storage).map(+), shape: lhs.shape)
    }

    let attentionNormalized = try normalize(
      state.residual, gamma: weights.attentionNormGamma)
    let queryProjection = try project(attentionNormalized, weights: weights.queryWeights)
    let keyProjection = try project(attentionNormalized, weights: weights.keyWeights)
    let valueProjection = try project(attentionNormalized, weights: weights.valueWeights)
    let queries = try reshapeHeads(queryProjection, headCount: configuration.queryHeadCount)
    let keys = try reshapeHeads(keyProjection, headCount: configuration.keyValueHeadCount)
    let values = try reshapeHeads(valueProjection, headCount: configuration.keyValueHeadCount)
    let rotatedQueries = try rotate(queries)
    let rotatedKeys = try rotate(keys)
    let attentionHeads = try causalAttention(
      queries: rotatedQueries, keys: rotatedKeys, values: values)
    let concatenatedAttention = try FloatTensor(
      attentionHeads.storage, shape: [sequenceLength, configuration.modelDimension])
    let attentionProjection = try project(
      concatenatedAttention, weights: weights.attentionOutputWeights)
    let postAttentionResidual = try add(state.residual, attentionProjection)
    let mlpNormalized = try normalize(postAttentionResidual, gamma: weights.mlpNormGamma)
    let gateProjection = try project(mlpNormalized, weights: weights.gateWeights)
    let upProjection = try project(mlpNormalized, weights: weights.upWeights)
    let activatedGate = try FloatTensor(
      gateProjection.storage.map { $0 / (1 + exp(-$0)) },
      shape: gateProjection.shape)
    let gatedHidden = try FloatTensor(
      zip(activatedGate.storage, upProjection.storage).map(*),
      shape: gateProjection.shape)
    let downProjection = try project(gatedHidden, weights: weights.downWeights)
    let finalResidual = try add(postAttentionResidual, downProjection)

    return DecoderBlockResult(
      state: DecoderBlockState(
        residual: finalResidual, positionOffset: state.positionOffset),
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
}