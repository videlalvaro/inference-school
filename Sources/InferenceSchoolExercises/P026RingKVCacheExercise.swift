import InferenceSchoolCore

public enum P026RingKVCacheExercise {
  public static func run(_ request: RingKVCacheRequest) throws -> RingKVCacheResult {
    let positions = Array(request.firstLogicalPosition...request.queryLogicalPosition)
    let history = positions.indices.map { Array(positions[0...$0]) }
    return RingKVCacheResult(
      chronologicalHistory: history,
      finalSnapshot: KVCacheLayerSnapshot(
        logicalPositions: positions,
        keys: request.keys,
        values: request.values),
      attentionOutput: try FloatTensor(
        Array(repeating: 0, count: request.query.elementCount), shape: request.query.shape),
      allocatedBytes: request.configuration.allocatedFloat32Bytes,
      storageAddressesStable: true)
  }
}
