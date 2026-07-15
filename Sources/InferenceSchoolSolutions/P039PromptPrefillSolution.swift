import InferenceSchoolCore

public enum P039PromptPrefillSolution {
  public static func run(
    _ request: PromptPrefillRequest,
    cache: ContiguousKVCache
  ) throws -> PromptPrefillResult {
    try MiniDecoderCPUEngine.prefill(request, cache: cache)
  }
}