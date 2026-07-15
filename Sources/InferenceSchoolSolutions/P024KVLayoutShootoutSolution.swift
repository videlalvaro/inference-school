import Foundation
import InferenceSchoolCore

public enum P024KVLayoutShootoutSolution {
  public static func run(
    logicalValues: FloatTensor,
    configuration: KVCacheConfiguration,
    layer: Int,
    head: Int
  ) throws -> KVLayoutShootoutResult {
    let expectedShape = [
      configuration.layerCount, configuration.capacity,
      configuration.keyValueHeadCount, configuration.headDimension,
    ]
    guard logicalValues.shape == expectedShape else {
      throw KVCacheError.vectorShapeMismatch(
        name: "Logical KV values", expected: expectedShape, actual: logicalValues.shape)
    }
    try configuration.validate(layer: layer)
    try configuration.validate(head: head)
    let tokenDescriptor = KVLayoutDescriptor(kind: .tokenMajor, configuration: configuration)
    let headDescriptor = KVLayoutDescriptor(kind: .headMajor, configuration: configuration)
    let tokenStorage = try copy(logicalValues, descriptor: tokenDescriptor)
    let headStorage = try copy(logicalValues, descriptor: headDescriptor)
    return KVLayoutShootoutResult(
      tokenMajorRoundTrip: try roundTrip(tokenStorage, descriptor: tokenDescriptor),
      headMajorRoundTrip: try roundTrip(headStorage, descriptor: headDescriptor),
      tokenMajorTrace: try trace(descriptor: tokenDescriptor, layer: layer, head: head),
      headMajorTrace: try trace(descriptor: headDescriptor, layer: layer, head: head))
  }

  public static func benchmark(
    configuration: KVCacheConfiguration,
    iterations: Int
  ) throws -> KVLayoutBenchmarkReport {
    guard iterations > 0 else { throw KVCacheError.invalidTokenCount(iterations) }
    let logical = try FloatTensor(
      (0..<configuration.elementsPerTensor).map { Float(($0 % 29) - 14) / 15 },
      shape: [configuration.layerCount, configuration.capacity,
        configuration.keyValueHeadCount, configuration.headDimension])
    let token = KVLayoutDescriptor(kind: .tokenMajor, configuration: configuration)
    let head = KVLayoutDescriptor(kind: .headMajor, configuration: configuration)
    let tokenStorage = try copy(logical, descriptor: token)
    let headStorage = try copy(logical, descriptor: head)
    let tokenMeasurement = try measure(
      storage: tokenStorage, descriptor: token, iterations: iterations)
    let headMeasurement = try measure(
      storage: headStorage, descriptor: head, iterations: iterations)
    return KVLayoutBenchmarkReport(
      configuration: configuration,
      iterations: iterations,
      tokenMajorNanoseconds: tokenMeasurement.nanoseconds,
      headMajorNanoseconds: headMeasurement.nanoseconds,
      checksum: tokenMeasurement.checksum + headMeasurement.checksum)
  }

  private static func copy(
    _ logical: FloatTensor, descriptor: KVLayoutDescriptor
  ) throws -> [Float] {
    let c = descriptor.configuration
    var storage = Array(repeating: Float.zero, count: c.elementsPerTensor)
    for layer in 0..<c.layerCount {
      for slot in 0..<c.capacity {
        for head in 0..<c.keyValueHeadCount {
          for feature in 0..<c.headDimension {
            let logicalOffset = (((layer * c.capacity + slot) * c.keyValueHeadCount + head)
              * c.headDimension + feature)
            storage[try descriptor.offset(
              layer: layer, slot: slot, head: head, feature: feature)] = logical.storage[logicalOffset]
          }
        }
      }
    }
    return storage
  }

  private static func roundTrip(
    _ storage: [Float], descriptor: KVLayoutDescriptor
  ) throws -> FloatTensor {
    let c = descriptor.configuration
    var logical = Array(repeating: Float.zero, count: c.elementsPerTensor)
    for layer in 0..<c.layerCount {
      for slot in 0..<c.capacity {
        for head in 0..<c.keyValueHeadCount {
          for feature in 0..<c.headDimension {
            let logicalOffset = (((layer * c.capacity + slot) * c.keyValueHeadCount + head)
              * c.headDimension + feature)
            logical[logicalOffset] = storage[try descriptor.offset(
              layer: layer, slot: slot, head: head, feature: feature)]
          }
        }
      }
    }
    return try FloatTensor(
      logical,
      shape: [c.layerCount, c.capacity, c.keyValueHeadCount, c.headDimension])
  }

  private static func trace(
    descriptor: KVLayoutDescriptor, layer: Int, head: Int
  ) throws -> KVAccessTrace {
    var offsets: [Int] = []
    for slot in 0..<descriptor.configuration.capacity {
      for feature in 0..<descriptor.configuration.headDimension {
        offsets.append(try descriptor.offset(
          layer: layer, slot: slot, head: head, feature: feature))
      }
    }
    let spans = offsets.enumerated().reduce(0) { count, item in
      item.offset == 0 || item.element != offsets[item.offset - 1] + 1 ? count + 1 : count
    }
    return KVAccessTrace(
      offsets: offsets,
      contiguousReadSpans: spans,
      bytesRead: offsets.count * MemoryLayout<Float>.stride)
  }

  private static func measure(
    storage: [Float],
    descriptor: KVLayoutDescriptor,
    iterations: Int
  ) throws -> (nanoseconds: UInt64, checksum: Float) {
    var samples: [UInt64] = []
    var checksum: Float = 0
    for _ in 0..<iterations {
      let start = DispatchTime.now().uptimeNanoseconds
      for layer in 0..<descriptor.configuration.layerCount {
        for head in 0..<descriptor.configuration.keyValueHeadCount {
          for slot in 0..<descriptor.configuration.capacity {
            for feature in 0..<descriptor.configuration.headDimension {
              checksum += storage[try descriptor.offset(
                layer: layer, slot: slot, head: head, feature: feature)]
            }
          }
        }
      }
      samples.append(DispatchTime.now().uptimeNanoseconds - start)
    }
    samples.sort()
    return (samples[samples.count / 2], checksum)
  }
}
