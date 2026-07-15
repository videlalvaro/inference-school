import InferenceSchoolCore

public final class PagedKVCache: KVCacheWritable {
  public let configuration: KVCacheConfiguration
  public let pageSize: Int
  public let physicalPageCount: Int
  public var allocatedBytes: Int {
    physicalPageCount * pageSize * configuration.elementsPerToken * 2
      * MemoryLayout<Float>.stride
  }

  private var keyStorage: [Float]
  private var valueStorage: [Float]
  private var pageTables: [[Int]]
  private var freePages: [Int]
  private var counts: [Int]
  private var firstPositions: [Int?]
  private var lastPositions: [Int?]

  public init(
    configuration: KVCacheConfiguration,
    pageSize: Int,
    physicalPageCount: Int
  ) throws {
    guard pageSize > 0 else { throw KVCacheError.invalidPageSize(pageSize) }
    guard physicalPageCount > 0 else { throw KVCacheError.invalidPageCount(physicalPageCount) }
    self.configuration = configuration
    self.pageSize = pageSize
    self.physicalPageCount = physicalPageCount
    let elementCount = physicalPageCount * pageSize * configuration.elementsPerToken
    keyStorage = Array(repeating: 0, count: elementCount)
    valueStorage = Array(repeating: 0, count: elementCount)
    pageTables = Array(repeating: [], count: configuration.layerCount)
    freePages = Array((0..<physicalPageCount).reversed())
    counts = Array(repeating: 0, count: configuration.layerCount)
    firstPositions = Array(repeating: nil, count: configuration.layerCount)
    lastPositions = Array(repeating: nil, count: configuration.layerCount)
  }

  public func count(layer: Int) throws -> Int {
    try configuration.validate(layer: layer)
    return counts[layer]
  }

  public func logicalPositions(layer: Int) throws -> [Int] {
    try configuration.validate(layer: layer)
    guard let first = firstPositions[layer] else { return [] }
    return (0..<counts[layer]).map { first + $0 }
  }

  public func physicalPages(layer: Int) throws -> [Int] {
    try configuration.validate(layer: layer)
    return pageTables[layer]
  }

  public func append(
    layer: Int,
    logicalPosition: Int,
    key: FloatTensor,
    value: FloatTensor
  ) throws {
    try configuration.validate(layer: layer)
    guard logicalPosition >= 0 else { throw KVCacheError.invalidLogicalPosition(logicalPosition) }
    try configuration.validate(vector: key, name: "Key")
    try configuration.validate(vector: value, name: "Value")
    guard counts[layer] < configuration.capacity else {
      throw KVCacheError.capacityExceeded(layer: layer, capacity: configuration.capacity)
    }
    if let last = lastPositions[layer], logicalPosition != last + 1 {
      throw KVCacheError.positionSequenceMismatch(
        layer: layer, expected: last + 1, actual: logicalPosition)
    }
    let logicalSlot = counts[layer]
    if logicalSlot.isMultiple(of: pageSize) {
      guard let physicalPage = freePages.popLast() else { throw KVCacheError.pageUnavailable }
      pageTables[layer].append(physicalPage)
    }
    let pageIndex = logicalSlot / pageSize
    let slotInPage = logicalSlot % pageSize
    let physicalPage = pageTables[layer][pageIndex]
    let start = ((physicalPage * pageSize + slotInPage) * configuration.elementsPerToken)
    for index in 0..<configuration.elementsPerToken {
      keyStorage[start + index] = key.storage[index]
      valueStorage[start + index] = value.storage[index]
    }
    if firstPositions[layer] == nil { firstPositions[layer] = logicalPosition }
    lastPositions[layer] = logicalPosition
    counts[layer] += 1
  }

  public func free(layer: Int) throws {
    try configuration.validate(layer: layer)
    for page in pageTables[layer] { freePages.append(page) }
    pageTables[layer].removeAll(keepingCapacity: true)
    counts[layer] = 0
    firstPositions[layer] = nil
    lastPositions[layer] = nil
  }

  public func keyVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try vector(storage: keyStorage, layer: layer, logicalPosition: logicalPosition, head: head)
  }

  public func valueVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try vector(storage: valueStorage, layer: layer, logicalPosition: logicalPosition, head: head)
  }

  public func allocatorReport() -> PageAllocatorReport {
    let allocated = physicalPageCount - freePages.count
    let liveTokens = counts.reduce(0, +)
    let sortedFree = freePages.sorted()
    var largestRun = 0
    var currentRun = 0
    var previous: Int?
    for page in sortedFree {
      currentRun = previous.map { page == $0 + 1 ? currentRun + 1 : 1 } ?? 1
      largestRun = max(largestRun, currentRun)
      previous = page
    }
    return PageAllocatorReport(
      physicalPageCount: physicalPageCount,
      allocatedPageCount: allocated,
      freePageCount: freePages.count,
      liveTokenCount: liveTokens,
      internalFragmentSlots: allocated * pageSize - liveTokens,
      largestContiguousFreeRun: largestRun)
  }

  private func vector(
    storage: [Float], layer: Int, logicalPosition: Int, head: Int
  ) throws -> [Float] {
    try configuration.validate(layer: layer)
    try configuration.validate(head: head)
    guard let first = firstPositions[layer] else {
      throw KVCacheError.positionNotCached(layer: layer, logicalPosition: logicalPosition)
    }
    let logicalSlot = logicalPosition - first
    guard logicalSlot >= 0, logicalSlot < counts[layer] else {
      throw KVCacheError.positionNotCached(layer: layer, logicalPosition: logicalPosition)
    }
    let physicalPage = pageTables[layer][logicalSlot / pageSize]
    let slotInPage = logicalSlot % pageSize
    let start = ((physicalPage * pageSize + slotInPage) * configuration.keyValueHeadCount + head)
      * configuration.headDimension
    return Array(storage[start..<(start + configuration.headDimension)])
  }
}

public enum P027PagedKVCacheSolution {
  public static func run(_ request: PagedKVCacheRequest) throws -> PagedKVCacheResult {
    let cache = try PagedKVCache(
      configuration: request.configuration,
      pageSize: request.pageSize,
      physicalPageCount: request.physicalPageCount)
    for operation in request.operations {
      switch operation {
      case .append(let layer, let position, let key, let value):
        try cache.append(
          layer: layer, logicalPosition: position, key: key, value: value)
      case .free(let layer):
        try cache.free(layer: layer)
      }
    }
    var snapshots: [KVCacheLayerSnapshot] = []
    var tables: [[Int]] = []
    for layer in 0..<request.configuration.layerCount {
      let materialized = try cache.materialized(layer: layer)
      snapshots.append(KVCacheLayerSnapshot(
        logicalPositions: materialized.positions,
        keys: materialized.keys,
        values: materialized.values))
      tables.append(try cache.physicalPages(layer: layer))
    }
    return PagedKVCacheResult(
      layerSnapshots: snapshots,
      physicalPageTables: tables,
      allocatorReport: cache.allocatorReport(),
      attentionOutput: try P023CachedAttentionSolution.attend(
        query: request.query,
        cache: cache,
        layer: request.attentionLayer,
        queryLogicalPosition: request.queryLogicalPosition,
        queryHeadCount: request.queryHeadCount),
      allocatedBytes: cache.allocatedBytes)
  }
}
