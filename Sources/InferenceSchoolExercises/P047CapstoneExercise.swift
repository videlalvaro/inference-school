import InferenceSchoolCore

public enum P047CapstoneExercise {
  public static func run(_ request: CapstoneRequest) throws -> CapstoneReport {
    try P047CapstoneContract.validate(request)
    // TODO: connect tokenizer, prefill, KV cache, serial decode, sampling, rendering, and report.
    return CapstoneReport(
      prompt: request.prompt,
      promptTokenIDs: [],
      generatedTokenIDs: [],
      generatedBytes: [],
      rendering: .text(""),
      stopReason: .maximumTokenCount,
      timings: [],
      timeToFirstTokenNanoseconds: nil,
      decodeTokensPerSecond: nil,
      finalCacheCounts: [],
      modelWeightBytes: 0,
      allocatedKVCacheBytes: 0,
      prefillArenaBytes: 0,
      decodeArenaBytes: 0,
      generationBackend: P047CapstoneContract.generationBackend,
      weightFormat: "Float32 row-major",
      keyValueFormat: "Float32 contiguous KV cache",
      metalVerification: CapstoneMetalVerification(
        label: P047CapstoneContract.metalVerificationLabel,
        status: .notRequested,
        captures: [],
        resources: nil),
      optimizationComparison: CapstoneOptimizationComparison(
        name: "not implemented",
        baselineDispatchCount: 1,
        optimizedDispatchCount: 1,
        baselineLogicalBytes: 0,
        optimizedLogicalBytes: 0,
        basis: "not implemented"),
      rejectedOptimization: CapstoneRejectedOptimization(
        name: "not implemented", evidence: "not implemented"),
      limitations: [])
  }
}
