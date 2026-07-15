import InferenceSchoolCore

public enum P022ContiguousKVCacheExercise {
  public static func run(
    configuration: KVCacheConfiguration,
    appends: [KVCacheAppend]
  ) throws -> ContiguousKVCacheTranscript {
    var counts = Array(repeating: 0, count: configuration.layerCount)
    var lastPositions = Array<Int?>(repeating: nil, count: configuration.layerCount)
    for append in appends {
      try P022ContiguousKVCacheContract.validate(
        append, configuration: configuration, counts: counts, lastPositions: lastPositions)
      counts[append.layer] += 1
      lastPositions[append.layer] = append.logicalPosition
    }

    let emptyLayers = try counts.map { _ in
      try KVCacheLayerSnapshot(
        logicalPositions: [],
        keys: FloatTensor([], shape: [0, configuration.keyValueHeadCount, configuration.headDimension]),
        values: FloatTensor([], shape: [0, configuration.keyValueHeadCount, configuration.headDimension]))
    }
    return ContiguousKVCacheTranscript(
      layers: emptyLayers,
      allocatedBytes: 0,
      keyStorageCount: 0,
      valueStorageCount: 0,
      storageAddressesStable: false)
  }
}
