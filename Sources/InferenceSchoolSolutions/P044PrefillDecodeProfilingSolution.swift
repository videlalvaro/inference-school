import Dispatch
import InferenceSchoolCore

public enum P044PrefillDecodeProfilingSolution {
  public static func profile(
    _ request: PrefillDecodeProfilingRequest
  ) throws -> PrefillDecodeProfileReport {
    try P044ProfilingContract.validate(request)

    for _ in 0..<request.warmupIterations {
      _ = try runFreshPrefill(
        model: request.model,
        tokenIDs: request.promptTokenIDs,
        capacity: request.promptTokenIDs.count)
    }

    var prefillSamples: [UInt64] = []
    var prefillWork: MiniDecoderWorkModel?
    for _ in 0..<request.measuredTrials {
      let start = DispatchTime.now().uptimeNanoseconds
      let result = try runFreshPrefill(
        model: request.model,
        tokenIDs: request.promptTokenIDs,
        capacity: request.promptTokenIDs.count)
      prefillSamples.append(positiveElapsed(since: start))
      prefillWork = result.work
    }
    let prefillLatency = try P044LatencyStatistics.summarize(
      prefillSamples, percentile: request.percentile)
    let prefillRate = Double(request.promptTokenIDs.count) * 1_000_000_000
      / prefillLatency.medianNanoseconds

    var decodeProfiles: [DecodeContextProfile] = []
    for contextLength in request.decodeContextLengths {
      for warmup in 0..<request.warmupIterations {
        _ = try runDecodeTrial(
          request: request,
          contextLength: contextLength,
          trialSeed: request.seed &+ UInt64(warmup),
          recordSamples: false)
      }

      var samples: [UInt64] = []
      var totalFLOPs = 0
      var totalWeightBytes = 0
      var totalCacheBytes = 0
      for trial in 0..<request.measuredTrials {
        let trialResult = try runDecodeTrial(
          request: request,
          contextLength: contextLength,
          trialSeed: request.seed &+ UInt64(trial),
          recordSamples: true)
        samples.append(contentsOf: trialResult.samples)
        for work in trialResult.work {
          totalFLOPs += work.projectionFLOPs + work.attentionFLOPs
          totalWeightBytes += work.estimatedWeightBytesRead
          totalCacheBytes += work.cacheBytesWritten
        }
      }
      let latency = try P044LatencyStatistics.summarize(
        samples, percentile: request.percentile)
      let measuredTokens = request.measuredTrials * request.decodeStepsPerTrial
      decodeProfiles.append(DecodeContextProfile(
        stageName: "decode.context.\(contextLength)",
        initialContextLength: contextLength,
        measuredTokenCount: measuredTokens,
        perTokenLatency: latency,
        decodeTokensPerSecond: 1_000_000_000 / latency.medianNanoseconds,
        averageWorkPerToken: ProfilingWorkEstimate(
          floatingPointOperations: totalFLOPs / measuredTokens,
          estimatedWeightBytesRead: totalWeightBytes / measuredTokens,
          cacheBytesWritten: totalCacheBytes / measuredTokens)))
    }

    let report = PrefillDecodeProfileReport(
      backend: P044ProfilingContract.cpuReferenceBackend,
      clock: "DispatchTime.uptimeNanoseconds (monotonic)",
      timingBoundary:
        "Prefill includes fresh ContiguousKVCache allocation and the engine call. Decode excludes prefill, but includes per-step tensor allocations, cache append, sampling, and host execution.",
      warmupIterations: request.warmupIterations,
      measuredTrials: request.measuredTrials,
      prefill: PrefillProfile(
        stageName: "prefill.ttft",
        promptTokenCount: request.promptTokenIDs.count,
        latency: prefillLatency,
        promptTokensPerSecond: prefillRate,
        work: ProfilingWorkEstimate(prefillWork!)),
      decode: decodeProfiles)
    try P044ProfilingContract.validate(report, for: request)
    return report
  }

  private struct DecodeTrialResult {
    let samples: [UInt64]
    let work: [MiniDecoderWorkModel]
  }

  private static func runFreshPrefill(
    model: MiniDecoderModel,
    tokenIDs: [Int],
    capacity: Int
  ) throws -> PromptPrefillResult {
    let cache = ContiguousKVCache(
      configuration: try model.cacheConfiguration(capacity: capacity))
    return try MiniDecoderCPUEngine.prefill(
      PromptPrefillRequest(model: model, tokenIDs: tokenIDs),
      cache: cache)
  }

  private static func runDecodeTrial(
    request: PrefillDecodeProfilingRequest,
    contextLength: Int,
    trialSeed: UInt64,
    recordSamples: Bool
  ) throws -> DecodeTrialResult {
    let prompt = (0..<contextLength).map {
      request.promptTokenIDs[$0 % request.promptTokenIDs.count]
    }
    let capacity = contextLength + request.decodeStepsPerTrial
    let cache = ContiguousKVCache(
      configuration: try request.model.cacheConfiguration(capacity: capacity))
    let prefill = try MiniDecoderCPUEngine.prefill(
      PromptPrefillRequest(model: request.model, tokenIDs: prompt),
      cache: cache)
    var sampler = SeededGenerator(seed: trialSeed)
    var greedyGenerator = SeededGenerator(seed: trialSeed)
    var tokenID = try P038LogitsSamplingSolution.sample(
      logits: prefill.logits.storage,
      strategy: .greedy,
      generator: &greedyGenerator).selectedToken
    var samples: [UInt64] = []
    var work: [MiniDecoderWorkModel] = []
    for step in 0..<request.decodeStepsPerTrial {
      let decodeRequest = AutoregressiveDecodeRequest(
        model: request.model,
        tokenID: tokenID,
        logicalPosition: contextLength + step,
        samplingStrategy: .greedy)
      let start = DispatchTime.now().uptimeNanoseconds
      let result = try MiniDecoderCPUEngine.decode(
        decodeRequest, cache: cache, generator: &sampler)
      let elapsed = positiveElapsed(since: start)
      if recordSamples {
        samples.append(elapsed)
        work.append(result.work)
      }
      tokenID = result.selectedNextTokenID
    }
    return DecodeTrialResult(samples: samples, work: work)
  }

  private static func positiveElapsed(since start: UInt64) -> UInt64 {
    max(1, DispatchTime.now().uptimeNanoseconds - start)
  }
}