import InferenceSchoolCore

public enum P022ContiguousKVCacheSolution {
  public static func run(
    configuration: KVCacheConfiguration,
    appends: [KVCacheAppend]
  ) throws -> ContiguousKVCacheTranscript {
    let cache = ContiguousKVCache(configuration: configuration)
    let before = cache.storageAddresses()
    for append in appends {
      try cache.append(
        layer: append.layer,
        logicalPosition: append.logicalPosition,
        key: append.key,
        value: append.value)
    }
    let after = cache.storageAddresses()
    var layers: [KVCacheLayerSnapshot] = []
    layers.reserveCapacity(configuration.layerCount)
    for layer in 0..<configuration.layerCount {
      let materialized = try cache.materialized(layer: layer)
      layers.append(KVCacheLayerSnapshot(
        logicalPositions: materialized.positions,
        keys: materialized.keys,
        values: materialized.values))
    }
    return ContiguousKVCacheTranscript(
      layers: layers,
      allocatedBytes: cache.allocatedBytes,
      keyStorageCount: cache.keyStorageCount,
      valueStorageCount: cache.valueStorageCount,
      storageAddressesStable: before == after)
  }
}
