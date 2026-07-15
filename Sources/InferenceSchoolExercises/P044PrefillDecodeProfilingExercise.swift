import InferenceSchoolCore

public enum P044PrefillDecodeProfilingExercise {
  public static func profile(
    _ request: PrefillDecodeProfilingRequest
  ) throws -> PrefillDecodeProfileReport {
    try P044ProfilingContract.validate(request)
    let zeroLatency = LatencyStatistics(
      samplesNanoseconds: Array(repeating: 0, count: request.measuredTrials),
      medianNanoseconds: 0,
      percentile: request.percentile,
      percentileNanoseconds: 0,
      minimumNanoseconds: 0,
      maximumNanoseconds: 0)
    // TODO: collect fresh-cache prefill trials and serial decode trials independently.
    return PrefillDecodeProfileReport(
      backend: P044ProfilingContract.cpuReferenceBackend,
      clock: "not implemented",
      timingBoundary: "not implemented",
      warmupIterations: request.warmupIterations,
      measuredTrials: request.measuredTrials,
      prefill: PrefillProfile(
        stageName: "prefill.ttft",
        promptTokenCount: request.promptTokenIDs.count,
        latency: zeroLatency,
        promptTokensPerSecond: 0,
        work: ProfilingWorkEstimate(
          floatingPointOperations: 0, estimatedWeightBytesRead: 0, cacheBytesWritten: 0)),
      decode: [])
  }
}