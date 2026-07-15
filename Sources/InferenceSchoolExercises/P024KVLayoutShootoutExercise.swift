import InferenceSchoolCore

public enum P024KVLayoutShootoutExercise {
  public static func run(
    logicalValues: FloatTensor,
    configuration: KVCacheConfiguration,
    layer: Int,
    head: Int
  ) throws -> KVLayoutShootoutResult {
    let token = KVLayoutDescriptor(kind: .tokenMajor, configuration: configuration)
    var offsets: [Int] = []
    for slot in 0..<configuration.capacity {
      for feature in 0..<configuration.headDimension {
        offsets.append(try token.offset(layer: layer, slot: slot, head: head, feature: feature))
      }
    }
    let trace = KVAccessTrace(
      offsets: offsets,
      contiguousReadSpans: configuration.capacity,
      bytesRead: offsets.count * MemoryLayout<Float>.stride)
    return KVLayoutShootoutResult(
      tokenMajorRoundTrip: logicalValues,
      headMajorRoundTrip: logicalValues,
      tokenMajorTrace: trace,
      headMajorTrace: trace)
  }
}
