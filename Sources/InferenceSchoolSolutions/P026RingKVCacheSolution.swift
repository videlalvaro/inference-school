import InferenceSchoolCore

public final class RingKVCache: KVCacheWritable {
  public let configuration: KVCacheConfiguration
  public var allocatedBytes: Int { configuration.allocatedFloat32Bytes }

  private var keyStorage: [Float]
  private var valueStorage: [Float]
  private var positions: [Int]
  private var counts: [Int]
  private var nextSlots: [Int]
  private var lastPositions: [Int?]

  public init(configuration: KVCacheConfiguration) {
    self.configuration = configuration
    keyStorage = Array(repeating: 0, count: configuration.elementsPerTensor)
    valueStorage = Array(repeating: 0, count: configuration.elementsPerTensor)
    positions = Array(repeating: -1, count: configuration.layerCount * configuration.capacity)
    counts = Array(repeating: 0, count: configuration.layerCount)
    nextSlots = Array(repeating: 0, count: configuration.layerCount)
    lastPositions = Array(repeating: nil, count: configuration.layerCount)
  }

  public func storageAddresses() -> (key: UInt, value: UInt) {
    (
      keyStorage.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) },
      valueStorage.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) }
    )
  }

  public func count(layer: Int) throws -> Int {
    try configuration.validate(layer: layer)
    return counts[layer]
  }

  public func logicalPositions(layer: Int) throws -> [Int] {
    try configuration.validate(layer: layer)
    let count = counts[layer]
    let oldestSlot = count == configuration.capacity ? nextSlots[layer] : 0
    return (0..<count).map { index in
      positions[layer * configuration.capacity + (oldestSlot + index) % configuration.capacity]
    }
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
    if let last = lastPositions[layer], logicalPosition != last + 1 {
      throw KVCacheError.positionSequenceMismatch(
        layer: layer, expected: last + 1, actual: logicalPosition)
    }
    let slot = nextSlots[layer]
    let start = offset(layer: layer, slot: slot, head: 0, feature: 0)
    for index in 0..<configuration.elementsPerToken {
      keyStorage[start + index] = key.storage[index]
      valueStorage[start + index] = value.storage[index]
    }
    positions[layer * configuration.capacity + slot] = logicalPosition
    counts[layer] = min(configuration.capacity, counts[layer] + 1)
    nextSlots[layer] = (slot + 1) % configuration.capacity
    lastPositions[layer] = logicalPosition
  }

  public func keyVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try vector(storage: keyStorage, layer: layer, logicalPosition: logicalPosition, head: head)
  }

  public func valueVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try vector(storage: valueStorage, layer: layer, logicalPosition: logicalPosition, head: head)
  }

  private func vector(
    storage: [Float], layer: Int, logicalPosition: Int, head: Int
  ) throws -> [Float] {
    try configuration.validate(layer: layer)
    try configuration.validate(head: head)
    guard let slot = (0..<configuration.capacity).first(where: {
      positions[layer * configuration.capacity + $0] == logicalPosition
    }) else {
      throw KVCacheError.positionNotCached(layer: layer, logicalPosition: logicalPosition)
    }
    let start = offset(layer: layer, slot: slot, head: head, feature: 0)
    return Array(storage[start..<(start + configuration.headDimension)])
  }

  private func offset(layer: Int, slot: Int, head: Int, feature: Int) -> Int {
    (((layer * configuration.capacity + slot) * configuration.keyValueHeadCount + head)
      * configuration.headDimension + feature)
  }
}

public enum P026RingKVCacheSolution {
  public static func run(_ request: RingKVCacheRequest) throws -> RingKVCacheResult {
    let cache = RingKVCache(configuration: request.configuration)
    let before = cache.storageAddresses()
    var history: [[Int]] = []
    for token in 0..<request.tokenCount {
      let start = token * request.configuration.elementsPerToken
      let end = start + request.configuration.elementsPerToken
      try cache.append(
        layer: request.layer,
        logicalPosition: request.firstLogicalPosition + token,
        key: FloatTensor(
          Array(request.keys.storage[start..<end]),
          shape: [request.configuration.keyValueHeadCount, request.configuration.headDimension]),
        value: FloatTensor(
          Array(request.values.storage[start..<end]),
          shape: [request.configuration.keyValueHeadCount, request.configuration.headDimension]))
      history.append(try cache.logicalPositions(layer: request.layer))
    }
    let materialized = try cache.materialized(layer: request.layer)
    return RingKVCacheResult(
      chronologicalHistory: history,
      finalSnapshot: KVCacheLayerSnapshot(
        logicalPositions: materialized.positions,
        keys: materialized.keys,
        values: materialized.values),
      attentionOutput: try P023CachedAttentionSolution.attend(
        query: request.query,
        cache: cache,
        layer: request.layer,
        queryLogicalPosition: request.queryLogicalPosition,
        queryHeadCount: request.queryHeadCount,
        window: request.window),
      allocatedBytes: cache.allocatedBytes,
      storageAddressesStable: before == cache.storageAddresses())
  }
}
