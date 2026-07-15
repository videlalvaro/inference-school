import InferenceSchoolCore

public enum P027PagedKVCacheExercise {
  public static func run(_ request: PagedKVCacheRequest) throws -> PagedKVCacheResult {
    let empty = try (0..<request.configuration.layerCount).map { _ in
      try KVCacheLayerSnapshot(
        logicalPositions: [],
        keys: FloatTensor(
          [], shape: [0, request.configuration.keyValueHeadCount,
            request.configuration.headDimension]),
        values: FloatTensor(
          [], shape: [0, request.configuration.keyValueHeadCount,
            request.configuration.headDimension]))
    }
    return PagedKVCacheResult(
      layerSnapshots: empty,
      physicalPageTables: Array(repeating: [], count: request.configuration.layerCount),
      allocatorReport: PageAllocatorReport(
        physicalPageCount: request.physicalPageCount,
        allocatedPageCount: 0,
        freePageCount: request.physicalPageCount,
        liveTokenCount: 0,
        internalFragmentSlots: 0,
        largestContiguousFreeRun: request.physicalPageCount),
      attentionOutput: try FloatTensor(
        Array(repeating: 0, count: request.query.elementCount), shape: request.query.shape),
      allocatedBytes: 0)
  }
}
