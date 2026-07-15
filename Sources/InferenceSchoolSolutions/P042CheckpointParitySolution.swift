import Foundation
import InferenceSchoolCore

public enum P042CheckpointParitySolution {
  public static func compare(
    _ request: CheckpointParityRequest
  ) throws -> CheckpointParityReport {
    let artifact = try P042CheckpointParityContract.validate(request)
    let candidate = try candidateCaptures(request)
    let candidateByName = Dictionary(
      uniqueKeysWithValues: candidate.captures.map { ($0.name, $0.tensor) })
    let referenceNames = Set(artifact.captures.map(\.name))
    if let unexpected = candidate.captures.first(where: {
      !referenceNames.contains($0.name)
    }) {
      throw CheckpointParityError.unexpectedCapture(unexpected.name)
    }

    var comparisons: [CaptureComparison] = []
    for reference in artifact.captures {
      guard let tensor = candidateByName[reference.name] else {
        throw CheckpointParityError.missingCapture(reference.name)
      }
      comparisons.append(compare(
        reference: reference,
        candidate: tensor,
        absoluteTolerance: request.absoluteTolerance,
        relativeTolerance: request.relativeTolerance))
    }
    let firstDivergence = comparisons.first(where: { !$0.passesTolerance })?.name
    let selectedTokenMatches = artifact.selectedTokenID == candidate.selectedTokenID
    return CheckpointParityReport(
      modelFingerprint: request.model.fingerprint,
      artifactProvenance: artifact.provenance,
      comparisons: comparisons,
      firstDivergentCapture: firstDivergence,
      referenceSelectedTokenID: artifact.selectedTokenID,
      candidateSelectedTokenID: candidate.selectedTokenID,
      selectedTokenMatches: selectedTokenMatches,
      isPassing: firstDivergence == nil && selectedTokenMatches)
  }

  private static func candidateCaptures(
    _ request: CheckpointParityRequest
  ) throws -> MiniDecoderCaptureSet {
    let model = request.model
    let sequence = request.tokenIDs.count
    let dimension = model.configuration.modelDimension
    var embedded: [Float] = []
    for tokenID in request.tokenIDs {
      let start = tokenID * dimension
      embedded.append(contentsOf: model.tokenEmbedding.storage[start..<(start + dimension)])
    }
    var residual = try FloatTensor(embedded, shape: [sequence, dimension])
    var traces: [MiniDecoderLayerTrace] = []
    for (layer, weights) in model.blocks.enumerated() {
      let input = residual
      let result: DecoderBlockResult
      switch request.fault {
      case .none:
        result = try P035DecoderBlockSolution.apply(
          state: DecoderBlockState(
            residual: residual, positionOffset: request.positionOffset),
          weights: weights,
          configuration: model.configuration)
      case .ropePositionOffset, .additiveRMSNormGamma:
        result = try faultedBlock(
          residual: residual,
          positionOffset: request.positionOffset,
          weights: weights,
          configuration: model.configuration,
          fault: request.fault)
      }
      residual = result.state.residual
      traces.append(MiniDecoderLayerTrace(
        layerIndex: layer,
        residualInput: input,
        block: result,
        cachePositions: Array(
          request.positionOffset..<(request.positionOffset + sequence))))
    }
    let finalNormalized = try MiniDecoderCPUEngine.normalize(
      residual,
      gamma: model.finalNormGamma,
      epsilon: model.configuration.rmsNormEpsilon,
      additiveGamma: request.fault == .additiveRMSNormGamma)
    let lastStart = (sequence - 1) * dimension
    let finalHidden = try FloatTensor(
      Array(finalNormalized.storage[lastStart..<(lastStart + dimension)]),
      shape: [dimension])
    let logits = try MiniDecoderCPUEngine.projectVector(
      finalHidden, weights: model.outputWeights)
    let selected = logits.storage.indices.dropFirst().reduce(0) { best, candidate in
      logits.storage[candidate] > logits.storage[best] ? candidate : best
    }
    let result = PromptPrefillResult(
      promptTokenCount: sequence,
      finalResidual: residual,
      finalNormalized: finalNormalized,
      finalHidden: finalHidden,
      logits: logits,
      layers: traces,
      cacheCounts: Array(repeating: sequence, count: model.layerCount),
      cachePositions: Array(
        repeating: Array(request.positionOffset..<(request.positionOffset + sequence)),
        count: model.layerCount),
      work: try MiniDecoderCPUEngine.workModel(
        model: model, tokenCount: sequence, decode: false))
    return try MiniDecoderCaptureSet.fromPrefill(
      result,
      model: model,
      tokenIDs: request.tokenIDs,
      positionOffset: request.positionOffset,
      selectedTokenID: selected)
  }

  private static func faultedBlock(
    residual: FloatTensor,
    positionOffset: Int,
    weights: DecoderBlockWeights,
    configuration: DecoderConfiguration,
    fault: MiniDecoderParityFault
  ) throws -> DecoderBlockResult {
    let sequence = residual.shape[0]
    let dimension = configuration.modelDimension
    let attentionNormalized = try MiniDecoderCPUEngine.normalize(
      residual,
      gamma: weights.attentionNormGamma,
      epsilon: configuration.rmsNormEpsilon,
      additiveGamma: fault == .additiveRMSNormGamma)
    let queryProjection = try MiniDecoderCPUEngine.project(
      attentionNormalized, weights: weights.queryWeights)
    let keyProjection = try MiniDecoderCPUEngine.project(
      attentionNormalized, weights: weights.keyWeights)
    let valueProjection = try MiniDecoderCPUEngine.project(
      attentionNormalized, weights: weights.valueWeights)
    let queries = try FloatTensor(
      queryProjection.storage,
      shape: [sequence, configuration.queryHeadCount, configuration.headDimension])
    let keys = try FloatTensor(
      keyProjection.storage,
      shape: [sequence, configuration.keyValueHeadCount, configuration.headDimension])
    let values = try FloatTensor(
      valueProjection.storage,
      shape: [sequence, configuration.keyValueHeadCount, configuration.headDimension])
    let ropeOffset = positionOffset + (fault == .ropePositionOffset ? 1 : 0)
    let rotatedQueries = try MiniDecoderCPUEngine.rotate(
      queries, positionOffset: ropeOffset, configuration: configuration)
    let rotatedKeys = try MiniDecoderCPUEngine.rotate(
      keys, positionOffset: ropeOffset, configuration: configuration)
    let attentionHeads = try groupedCausalAttention(
      queries: rotatedQueries,
      keys: rotatedKeys,
      values: values,
      configuration: configuration)
    let concatenated = try FloatTensor(attentionHeads.storage, shape: [sequence, dimension])
    let attentionProjection = try MiniDecoderCPUEngine.project(
      concatenated, weights: weights.attentionOutputWeights)
    let postAttention = try MiniDecoderCPUEngine.add(residual, attentionProjection)
    let mlpNormalized = try MiniDecoderCPUEngine.normalize(
      postAttention,
      gamma: weights.mlpNormGamma,
      epsilon: configuration.rmsNormEpsilon,
      additiveGamma: fault == .additiveRMSNormGamma)
    let gate = try MiniDecoderCPUEngine.project(mlpNormalized, weights: weights.gateWeights)
    let up = try MiniDecoderCPUEngine.project(mlpNormalized, weights: weights.upWeights)
    let activated = try FloatTensor(
      gate.storage.map { $0 / (1 + exp(-$0)) }, shape: gate.shape)
    let gated = try FloatTensor(zip(activated.storage, up.storage).map(*), shape: gate.shape)
    let down = try MiniDecoderCPUEngine.project(gated, weights: weights.downWeights)
    let output = try MiniDecoderCPUEngine.add(postAttention, down)
    return DecoderBlockResult(
      state: DecoderBlockState(residual: output, positionOffset: positionOffset),
      intermediates: DecoderBlockIntermediates(
        attentionNormalized: attentionNormalized,
        queries: queries,
        keys: keys,
        values: values,
        rotatedQueries: rotatedQueries,
        rotatedKeys: rotatedKeys,
        attentionHeads: attentionHeads,
        concatenatedAttention: concatenated,
        attentionProjection: attentionProjection,
        postAttentionResidual: postAttention,
        mlpNormalized: mlpNormalized,
        gateProjection: gate,
        upProjection: up,
        activatedGate: activated,
        gatedHidden: gated,
        downProjection: down))
  }

  private static func groupedCausalAttention(
    queries: FloatTensor,
    keys: FloatTensor,
    values: FloatTensor,
    configuration: DecoderConfiguration
  ) throws -> FloatTensor {
    let sequence = queries.shape[0]
    let headDimension = configuration.headDimension
    let groupSize = configuration.queryHeadCount / configuration.keyValueHeadCount
    let scale = 1 / sqrt(Float(headDimension))
    var output = Array(repeating: Float.zero, count: queries.elementCount)
    for query in 0..<sequence {
      for queryHead in 0..<configuration.queryHeadCount {
        let keyValueHead = queryHead / groupSize
        var scores = Array(repeating: Float.zero, count: query + 1)
        for key in 0...query {
          for feature in 0..<headDimension {
            scores[key] += queries.storage[
              (query * configuration.queryHeadCount + queryHead) * headDimension + feature]
              * keys.storage[
                (key * configuration.keyValueHeadCount + keyValueHead) * headDimension + feature]
          }
          scores[key] *= scale
        }
        let maximum = scores.max()!
        let exponentials = scores.map { exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        for feature in 0..<headDimension {
          for key in 0...query {
            output[(query * configuration.queryHeadCount + queryHead) * headDimension + feature]
              += exponentials[key] / denominator
                * values.storage[
                  (key * configuration.keyValueHeadCount + keyValueHead) * headDimension + feature]
          }
        }
      }
    }
    return try FloatTensor(output, shape: queries.shape)
  }

  private static func compare(
    reference: ReferenceCaptureTensor,
    candidate: FloatTensor,
    absoluteTolerance: Float,
    relativeTolerance: Float
  ) -> CaptureComparison {
    guard reference.shape == candidate.shape else {
      return CaptureComparison(
        name: reference.name,
        referenceShape: reference.shape,
        candidateShape: candidate.shape,
        maximumAbsoluteError: nil,
        rootMeanSquareError: nil,
        cosineSimilarity: nil,
        argmaxMatches: nil,
        passesTolerance: false)
    }
    var maximumError = 0.0
    var squareErrorSum = 0.0
    var dot = 0.0
    var referenceNorm = 0.0
    var candidateNorm = 0.0
    var passes = true
    for index in reference.values.indices {
      let expected = Double(reference.values[index])
      let actual = Double(candidate.storage[index])
      let error = abs(actual - expected)
      maximumError = max(maximumError, error)
      squareErrorSum += error * error
      dot += expected * actual
      referenceNorm += expected * expected
      candidateNorm += actual * actual
      let allowed = Double(absoluteTolerance) + Double(relativeTolerance) * abs(expected)
      if error > allowed { passes = false }
    }
    let count = max(reference.values.count, 1)
    let cosine: Double
    if referenceNorm == 0, candidateNorm == 0 {
      cosine = 1
    } else if referenceNorm == 0 || candidateNorm == 0 {
      cosine = 0
    } else {
      cosine = dot / sqrt(referenceNorm * candidateNorm)
    }
    let shouldCompareArgmax = reference.name == "logits" || reference.name == "selected_token"
    let referenceArgmax = reference.values.indices.dropFirst().reduce(0) {
      reference.values[$1] > reference.values[$0] ? $1 : $0
    }
    let candidateArgmax = candidate.storage.indices.dropFirst().reduce(0) {
      candidate.storage[$1] > candidate.storage[$0] ? $1 : $0
    }
    return CaptureComparison(
      name: reference.name,
      referenceShape: reference.shape,
      candidateShape: candidate.shape,
      maximumAbsoluteError: maximumError,
      rootMeanSquareError: sqrt(squareErrorSum / Double(count)),
      cosineSimilarity: cosine,
      argmaxMatches: shouldCompareArgmax ? referenceArgmax == candidateArgmax : nil,
      passesTolerance: passes)
  }
}