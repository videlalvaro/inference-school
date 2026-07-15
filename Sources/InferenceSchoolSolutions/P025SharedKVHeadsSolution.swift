import InferenceSchoolCore

public enum P025SharedKVHeadsSolution {
  public static func run(_ request: CachedAttentionRequest) throws -> SharedKVHeadsResult {
    SharedKVHeadsResult(
      attention: try P023CachedAttentionSolution.run(request),
      bytes: try KVHeadMemoryModel.compare(
        layerCount: request.cacheConfiguration.layerCount,
        tokenCount: request.cacheConfiguration.capacity,
        queryHeadCount: request.attentionConfiguration.queryHeadCount,
        gqaHeadCount: request.cacheConfiguration.keyValueHeadCount,
        headDimension: request.cacheConfiguration.headDimension))
  }
}
