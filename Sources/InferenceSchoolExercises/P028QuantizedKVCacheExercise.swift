import InferenceSchoolCore

public enum P028QuantizedKVCacheExercise {
  public static func run(_ request: CachedAttentionRequest) throws -> QuantizedKVCacheResult {
    let zeroOutput = try FloatTensor(
      Array(repeating: 0, count: request.query.elementCount), shape: request.query.shape)
    return QuantizedKVCacheResult(
      attentionOutput: zeroOutput,
      dequantizedKeys: request.keys,
      dequantizedValues: request.values,
      keyScales: [],
      valueScales: [],
      allocatedBytes: request.cacheConfiguration.allocatedFloat32Bytes,
      maximumKeyError: 0,
      maximumValueError: 0)
  }
}
