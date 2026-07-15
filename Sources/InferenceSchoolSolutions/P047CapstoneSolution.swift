import Dispatch
import InferenceSchoolCore
import Metal

public enum P047CapstoneSolution {
  public static func run(_ request: CapstoneRequest) throws -> CapstoneReport {
    try P047CapstoneContract.validate(request)
    let promptTokenIDs = try P037ByteBPESolution.encode(
      tokenizer: request.tokenizer,
      text: request.prompt,
      options: BPEEncodingOptions(addBeginningOfSequence: true))
    try request.model.validate(tokenIDs: promptTokenIDs)
    let roundTrip = try P037ByteBPESolution.decodeBytes(
      tokenizer: request.tokenizer,
      tokenIDs: promptTokenIDs,
      skipSpecialTokens: true)
    guard roundTrip == Array(request.prompt.utf8) else {
      throw CapstoneError.tokenizerRoundTripMismatch
    }

    let capacity = promptTokenIDs.count + max(1, request.maxNewTokens)
    let cache = ContiguousKVCache(
      configuration: try request.model.cacheConfiguration(capacity: capacity))
    var generator = SeededGenerator(seed: request.seed)
    let prefillStart = DispatchTime.now().uptimeNanoseconds
    let prefill = try MiniDecoderCPUEngine.prefill(
      PromptPrefillRequest(model: request.model, tokenIDs: promptTokenIDs),
      cache: cache)
    let prefillNanoseconds = positiveElapsed(since: prefillStart)
    var timings = [CapstoneStageTiming(name: "prefill.engine", nanoseconds: prefillNanoseconds)]
    var generated: [Int] = []
    var stopReason = GenerationStopReason.maximumTokenCount
    var timeToFirstToken: UInt64?

    if request.maxNewTokens > 0 {
      let firstStart = DispatchTime.now().uptimeNanoseconds
      let first = try P038LogitsSamplingSolution.sample(
        logits: prefill.logits.storage,
        strategy: request.samplingStrategy,
        generator: &generator)
      let firstSamplingNanoseconds = positiveElapsed(since: firstStart)
      generated.append(first.selectedToken)
      timeToFirstToken = prefillNanoseconds + firstSamplingNanoseconds
      if first.selectedToken == request.tokenizer.endOfSequenceTokenID {
        stopReason = .endOfSequence
      }
    }

    while generated.count < request.maxNewTokens, stopReason != .endOfSequence {
      let stepIndex = generated.count
      let start = DispatchTime.now().uptimeNanoseconds
      let result = try MiniDecoderCPUEngine.decode(
        AutoregressiveDecodeRequest(
          model: request.model,
          tokenID: generated.last!,
          logicalPosition: promptTokenIDs.count + stepIndex - 1,
          samplingStrategy: request.samplingStrategy),
        cache: cache,
        generator: &generator)
      let elapsed = positiveElapsed(since: start)
      timings.append(CapstoneStageTiming(
        name: "decode.token.\(stepIndex)", nanoseconds: elapsed))
      generated.append(result.selectedNextTokenID)
      if result.selectedNextTokenID == request.tokenizer.endOfSequenceTokenID {
        stopReason = .endOfSequence
      }
    }

    let generatedBytes = try P037ByteBPESolution.decodeBytes(
      tokenizer: request.tokenizer,
      tokenIDs: generated,
      skipSpecialTokens: true)
    let rendering: GeneratedRendering
    if let text = String(bytes: generatedBytes, encoding: .utf8) {
      rendering = .text(text)
    } else {
      rendering = .hexadecimal(generatedBytes.map { String(format: "%02x", $0) }.joined())
    }

    let finalCacheCounts = try (0..<request.model.layerCount).map {
      try cache.count(layer: $0)
    }
    let arena = try P041BufferPlanningSolution.compareDecoderPlans(
      model: request.model,
      prefillTokenCount: promptTokenIDs.count,
      cachedTokenCount: max(1, finalCacheCounts[0]))
    let layerZeroInput = prefill.layers[0].residualInput
    let weights = request.model.blocks[0]
    let fusedRequest = FusedQKVRequest(
      input: layerZeroInput,
      gamma: weights.attentionNormGamma,
      queryWeights: weights.queryWeights,
      keyWeights: weights.keyWeights,
      valueWeights: weights.valueWeights,
      epsilon: request.model.configuration.rmsNormEpsilon,
      configuration: request.model.configuration)
    let fusionCost = try P043FusedQKVCostModel.compare(fusedRequest)
    let metalVerification = try verifyMetalIfRequested(
      request: request,
      fusedRequest: fusedRequest)
    let decodeDurations = timings.filter { $0.name.hasPrefix("decode.token.") }
    let decodeRate: Double? = decodeDurations.isEmpty ? nil :
      Double(decodeDurations.count) * 1_000_000_000
        / Double(decodeDurations.reduce(0) { $0 + $1.nanoseconds })
    let rejectionEvidence: String
    if let resources = metalVerification.resources {
      rejectionEvidence =
        "The verification slice performs \(resources.commandBufferCount) command buffers, \(resources.allocatedBufferBytes) bytes of per-call buffer allocation, \(resources.hostToDeviceBytes) host-to-device bytes, \(resources.deviceToHostBytes) device-to-host bytes, and \(resources.hostWaitCount) synchronous waits; attention, MLP, sampling, and cache execution remain on CPU."
    } else {
      rejectionEvidence =
        "Only the CPU engine executes attention, MLP, sampling, and cache updates; no completed Metal verification was available to support labeling this a Metal inference backend."
    }

    let report = CapstoneReport(
      prompt: request.prompt,
      promptTokenIDs: promptTokenIDs,
      generatedTokenIDs: generated,
      generatedBytes: generatedBytes,
      rendering: rendering,
      stopReason: stopReason,
      timings: timings,
      timeToFirstTokenNanoseconds: timeToFirstToken,
      decodeTokensPerSecond: decodeRate,
      finalCacheCounts: finalCacheCounts,
      modelWeightBytes: modelWeightBytes(request.model),
      allocatedKVCacheBytes: cache.allocatedBytes,
      prefillArenaBytes: arena.prefill.arenaByteCount,
      decodeArenaBytes: arena.decode.arenaByteCount,
      generationBackend: P047CapstoneContract.generationBackend,
      weightFormat: "Float32 row-major",
      keyValueFormat: "Float32 contiguous KV cache",
      metalVerification: metalVerification,
      optimizationComparison: CapstoneOptimizationComparison(
        name: "RMSNorm + Q/K/V fusion",
        baselineDispatchCount: fusionCost.separate.dispatchCount,
        optimizedDispatchCount: fusionCost.fused.dispatchCount,
        baselineLogicalBytes: fusionCost.separate.logicalTensorBytes,
        optimizedLogicalBytes: fusionCost.fused.logicalTensorBytes,
        basis: "Modeled logical tensor traffic and dispatches for layer 0 over this prompt; not a wall-clock speedup claim."),
      rejectedOptimization: CapstoneRejectedOptimization(
        name: "Label the verification slice as a Metal inference backend",
        evidence: rejectionEvidence),
      limitations: [
        "The seven-token tokenizer is restricted to the bytes a, b, c, space, and period plus BOS/EOS.",
        "The deterministic educational fixture is not pretrained and its output is not a language-quality demonstration.",
        "Generation is batch size one on the CPU reference backend.",
        "Metal executes only fused QKV and RoPE verification captures, not the full decoder.",
        "Wall-clock timings are machine- and run-specific and include Swift tensor allocation overhead.",
      ])
    try P047CapstoneContract.validate(report, for: request)
    return report
  }

  private static func verifyMetalIfRequested(
    request: CapstoneRequest,
    fusedRequest: FusedQKVRequest
  ) throws -> CapstoneMetalVerification {
    guard request.includeMetalVerification else {
      return CapstoneMetalVerification(
        label: P047CapstoneContract.metalVerificationLabel,
        status: .notRequested,
        captures: [],
        resources: nil)
    }
    guard MTLCreateSystemDefaultDevice() != nil else {
      return CapstoneMetalVerification(
        label: P047CapstoneContract.metalVerificationLabel,
        status: .unavailable("No Metal device is available."),
        captures: [],
        resources: nil)
    }

    let cpuFused = try P043FusedQKVSolution.fused(fusedRequest)
    let fusedPipeline = try P043FusedQKVSolution.makeMetalPipeline()
    let metalFused = try fusedPipeline.run(fusedRequest)
    let configuration = request.model.configuration
    let cpuRoPE = try P015RoPESolution.apply(
      queries: cpuFused.queries,
      keys: cpuFused.keys,
      rotaryDimension: configuration.rotaryDimension,
      base: configuration.ropeBase,
      queryPositionOffset: 0,
      keyPositionOffset: 0)
    let ropePipeline = try P015RoPESolution.makeMetalPipeline()
    let metalRoPE = try ropePipeline.apply(
      metalFused.result.queries,
      metalFused.result.keys,
      rotaryDimension: configuration.rotaryDimension,
      base: configuration.ropeBase,
      queryPositionOffset: 0,
      keyPositionOffset: 0)
    let captures = [
      parity(name: "layer.0.fused_qkv.query", metalFused.result.queries, cpuFused.queries),
      parity(name: "layer.0.fused_qkv.key", metalFused.result.keys, cpuFused.keys),
      parity(name: "layer.0.fused_qkv.value", metalFused.result.values, cpuFused.values),
      parity(name: "layer.0.rope.query", metalRoPE.queries, cpuRoPE.queries),
      parity(name: "layer.0.rope.key", metalRoPE.keys, cpuRoPE.keys),
    ]
    let queryBytes = cpuFused.queries.elementCount * MemoryLayout<Float>.stride
    let keyBytes = cpuFused.keys.elementCount * MemoryLayout<Float>.stride
    let ropeAllocated = 2 * queryBytes + 2 * keyBytes
    return CapstoneMetalVerification(
      label: P047CapstoneContract.metalVerificationLabel,
      status: .completed,
      captures: captures,
      resources: MetalVerificationResources(
        allocatedBufferBytes: metalFused.allocatedBufferBytes + ropeAllocated,
        hostToDeviceBytes: metalFused.hostToDeviceBytes + ropeAllocated,
        deviceToHostBytes: metalFused.deviceToHostBytes + queryBytes + keyBytes,
        dispatchCount: metalFused.dispatchCount + 2,
        commandBufferCount: metalFused.commandBufferCount + 2,
        hostWaitCount: metalFused.hostWaitCount + 2))
  }

  private static func parity(
    name: String,
    _ actual: FloatTensor,
    _ expected: FloatTensor
  ) -> CapstoneParityCapture {
    let maximum = zip(actual.storage, expected.storage)
      .map { abs(Double($0) - Double($1)) }
      .max() ?? 0
    let passes = actual.shape == expected.shape
      && zip(actual.storage, expected.storage).allSatisfy { actual, expected in
        abs(actual - expected) <= 5e-5 + 1e-4 * max(abs(actual), abs(expected))
      }
    return CapstoneParityCapture(
      name: name, maximumAbsoluteError: maximum, passes: passes)
  }

  private static func modelWeightBytes(_ model: MiniDecoderModel) -> Int {
    let elements = model.tokenEmbedding.elementCount
      + model.finalNormGamma.elementCount
      + model.blocks.reduce(0) { partial, block in
        partial + block.attentionNormGamma.elementCount + block.queryWeights.elementCount
          + block.keyWeights.elementCount + block.valueWeights.elementCount
          + block.attentionOutputWeights.elementCount + block.mlpNormGamma.elementCount
          + block.gateWeights.elementCount + block.upWeights.elementCount
          + block.downWeights.elementCount
      }
      + (model.outputProjection == .tiedEmbedding ? 0 : model.outputWeights.elementCount)
    return elements * MemoryLayout<Float>.stride
  }

  private static func positiveElapsed(since start: UInt64) -> UInt64 {
    max(1, DispatchTime.now().uptimeNanoseconds - start)
  }
}
