import InferenceSchoolCore

public final class QuantizedKVCache: KVCacheWritable {
  public let configuration: KVCacheConfiguration
  public var allocatedBytes: Int {
    keyStorage.count * MemoryLayout<Int8>.stride
      + valueStorage.count * MemoryLayout<Int8>.stride
      + keyScales.count * MemoryLayout<Float>.stride
      + valueScales.count * MemoryLayout<Float>.stride
  }

  private var keyStorage: [Int8]
  private var valueStorage: [Int8]
  private var keyScales: [Float]
  private var valueScales: [Float]
  private var positions: [Int]
  private var counts: [Int]
  private var firstPositions: [Int?]

  public init(configuration: KVCacheConfiguration) {
    self.configuration = configuration
    keyStorage = Array(repeating: 0, count: configuration.elementsPerTensor)
    valueStorage = Array(repeating: 0, count: configuration.elementsPerTensor)
    let vectorCount = configuration.layerCount * configuration.capacity
      * configuration.keyValueHeadCount
    keyScales = Array(repeating: 1, count: vectorCount)
    valueScales = Array(repeating: 1, count: vectorCount)
    positions = Array(repeating: -1, count: configuration.layerCount * configuration.capacity)
    counts = Array(repeating: 0, count: configuration.layerCount)
    firstPositions = Array(repeating: nil, count: configuration.layerCount)
  }

  public func count(layer: Int) throws -> Int {
    try configuration.validate(layer: layer)
    return counts[layer]
  }

  public func logicalPositions(layer: Int) throws -> [Int] {
    try configuration.validate(layer: layer)
    let start = layer * configuration.capacity
    return Array(positions[start..<(start + counts[layer])])
  }

  public func append(
    layer: Int,
    logicalPosition: Int,
    key: FloatTensor,
    value: FloatTensor
  ) throws {
    let record = KVCacheAppend(
      layer: layer, logicalPosition: logicalPosition, key: key, value: value)
    let lastPositions = firstPositions.enumerated().map { index, first in
      first.map { $0 + counts[index] - 1 }
    }
    try P022ContiguousKVCacheContract.validate(
      record, configuration: configuration, counts: counts, lastPositions: lastPositions)
    let slot = counts[layer]
    for head in 0..<configuration.keyValueHeadCount {
      let source = head * configuration.headDimension
      let destination = elementOffset(layer: layer, slot: slot, head: head)
      let scaleIndex = vectorOffset(layer: layer, slot: slot, head: head)
      quantize(
        Array(key.storage[source..<(source + configuration.headDimension)]),
        into: &keyStorage,
        offset: destination,
        scales: &keyScales,
        scaleIndex: scaleIndex)
      quantize(
        Array(value.storage[source..<(source + configuration.headDimension)]),
        into: &valueStorage,
        offset: destination,
        scales: &valueScales,
        scaleIndex: scaleIndex)
    }
    positions[layer * configuration.capacity + slot] = logicalPosition
    if firstPositions[layer] == nil { firstPositions[layer] = logicalPosition }
    counts[layer] += 1
  }

  public func keyVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try dequantize(
      storage: keyStorage, scales: keyScales,
      layer: layer, logicalPosition: logicalPosition, head: head)
  }

  public func valueVector(layer: Int, logicalPosition: Int, head: Int) throws -> [Float] {
    try dequantize(
      storage: valueStorage, scales: valueScales,
      layer: layer, logicalPosition: logicalPosition, head: head)
  }

  public func rawKeyStorage() -> [Int8] { keyStorage }
  public func rawValueStorage() -> [Int8] { valueStorage }
  public func rawKeyScales() -> [Float] { keyScales }
  public func rawValueScales() -> [Float] { valueScales }

  private func quantize(
    _ values: [Float],
    into storage: inout [Int8],
    offset: Int,
    scales: inout [Float],
    scaleIndex: Int
  ) {
    let maximum = values.map { abs($0) }.max() ?? 0
    let scale = maximum == 0 ? 1 : maximum / 127
    scales[scaleIndex] = scale
    for index in values.indices {
      let rounded = (values[index] / scale).rounded()
      storage[offset + index] = Int8(max(-127, min(127, Int(rounded))))
    }
  }

  private func dequantize(
    storage: [Int8],
    scales: [Float],
    layer: Int,
    logicalPosition: Int,
    head: Int
  ) throws -> [Float] {
    try configuration.validate(layer: layer)
    try configuration.validate(head: head)
    guard let first = firstPositions[layer] else {
      throw KVCacheError.positionNotCached(layer: layer, logicalPosition: logicalPosition)
    }
    let slot = logicalPosition - first
    guard slot >= 0, slot < counts[layer],
      positions[layer * configuration.capacity + slot] == logicalPosition
    else { throw KVCacheError.positionNotCached(layer: layer, logicalPosition: logicalPosition) }
    let start = elementOffset(layer: layer, slot: slot, head: head)
    let scale = scales[vectorOffset(layer: layer, slot: slot, head: head)]
    return (0..<configuration.headDimension).map { Float(storage[start + $0]) * scale }
  }

  private func vectorOffset(layer: Int, slot: Int, head: Int) -> Int {
    (layer * configuration.capacity + slot) * configuration.keyValueHeadCount + head
  }

  private func elementOffset(layer: Int, slot: Int, head: Int) -> Int {
    vectorOffset(layer: layer, slot: slot, head: head) * configuration.headDimension
  }
}

public enum P028QuantizedKVCacheSolution {
  public static func run(_ request: CachedAttentionRequest) throws -> QuantizedKVCacheResult {
    let cache = try populatedCache(request)
    return try result(
      request: request,
      cache: cache,
      output: P023CachedAttentionSolution.attend(
        query: request.query,
        cache: cache,
        layer: request.layer,
        queryLogicalPosition: request.queryLogicalPosition,
        queryHeadCount: request.attentionConfiguration.queryHeadCount))
  }

  public static func runMetal(
    _ request: CachedAttentionRequest,
    pipeline: MetalQuantizedCachedAttentionPipeline
  ) throws -> QuantizedKVCacheResult {
    let cache = try populatedCache(request)
    return try result(
      request: request,
      cache: cache,
      output: pipeline.apply(
        query: request.query,
        keyStorage: cache.rawKeyStorage(),
        valueStorage: cache.rawValueStorage(),
        keyScales: cache.rawKeyScales(),
        valueScales: cache.rawValueScales(),
        configuration: cache.configuration,
        queryHeadCount: request.attentionConfiguration.queryHeadCount,
        layer: request.layer,
        tokenCount: request.tokenCount))
  }

  private static func populatedCache(_ request: CachedAttentionRequest) throws -> QuantizedKVCache {
    let cache = QuantizedKVCache(configuration: request.cacheConfiguration)
    let elements = request.cacheConfiguration.elementsPerToken
    for token in 0..<request.tokenCount {
      let start = token * elements
      let end = start + elements
      try cache.append(
        layer: request.layer,
        logicalPosition: request.firstLogicalPosition + token,
        key: FloatTensor(
          Array(request.keys.storage[start..<end]),
          shape: [request.cacheConfiguration.keyValueHeadCount,
            request.cacheConfiguration.headDimension]),
        value: FloatTensor(
          Array(request.values.storage[start..<end]),
          shape: [request.cacheConfiguration.keyValueHeadCount,
            request.cacheConfiguration.headDimension]))
    }
    return cache
  }

  private static func result(
    request: CachedAttentionRequest,
    cache: QuantizedKVCache,
    output: FloatTensor
  ) throws -> QuantizedKVCacheResult {
    let materialized = try cache.materialized(layer: request.layer)
    let keyErrors = zip(request.keys.storage, materialized.keys.storage).map { abs($0 - $1) }
    let valueErrors = zip(request.values.storage, materialized.values.storage).map { abs($0 - $1) }
    return QuantizedKVCacheResult(
      attentionOutput: output,
      dequantizedKeys: materialized.keys,
      dequantizedValues: materialized.values,
      keyScales: Array(cache.rawKeyScales().prefix(
        request.tokenCount * request.cacheConfiguration.keyValueHeadCount)),
      valueScales: Array(cache.rawValueScales().prefix(
        request.tokenCount * request.cacheConfiguration.keyValueHeadCount)),
      allocatedBytes: cache.allocatedBytes,
      maximumKeyError: keyErrors.max() ?? 0,
      maximumValueError: valueErrors.max() ?? 0)
  }
}
