import Foundation

public enum KVLayoutKind: String, CaseIterable, Sendable {
  case tokenMajor = "[L,T,H,D]"
  case headMajor = "[L,H,T,D]"
}

public struct KVLayoutDescriptor: Sendable, Equatable {
  public let kind: KVLayoutKind
  public let configuration: KVCacheConfiguration

  public init(kind: KVLayoutKind, configuration: KVCacheConfiguration) {
    self.kind = kind
    self.configuration = configuration
  }

  public func offset(layer: Int, slot: Int, head: Int, feature: Int) throws -> Int {
    try configuration.validate(layer: layer)
    try configuration.validate(head: head)
    guard slot >= 0, slot < configuration.capacity else {
      throw KVCacheError.slotOutOfBounds(slot: slot, capacity: configuration.capacity)
    }
    guard feature >= 0, feature < configuration.headDimension else {
      throw TensorError.indexOutOfBounds(
        axis: 3, index: feature, dimension: configuration.headDimension)
    }
    switch kind {
    case .tokenMajor:
      return (((layer * configuration.capacity + slot) * configuration.keyValueHeadCount + head)
        * configuration.headDimension + feature)
    case .headMajor:
      return (((layer * configuration.keyValueHeadCount + head) * configuration.capacity + slot)
        * configuration.headDimension + feature)
    }
  }
}

public struct KVAccessTrace: Sendable, Equatable {
  public let offsets: [Int]
  public let contiguousReadSpans: Int
  public let bytesRead: Int

  public init(offsets: [Int], contiguousReadSpans: Int, bytesRead: Int) {
    self.offsets = offsets
    self.contiguousReadSpans = contiguousReadSpans
    self.bytesRead = bytesRead
  }
}

public struct KVLayoutShootoutResult: Sendable, Equatable {
  public let tokenMajorRoundTrip: FloatTensor
  public let headMajorRoundTrip: FloatTensor
  public let tokenMajorTrace: KVAccessTrace
  public let headMajorTrace: KVAccessTrace

  public init(
    tokenMajorRoundTrip: FloatTensor,
    headMajorRoundTrip: FloatTensor,
    tokenMajorTrace: KVAccessTrace,
    headMajorTrace: KVAccessTrace
  ) {
    self.tokenMajorRoundTrip = tokenMajorRoundTrip
    self.headMajorRoundTrip = headMajorRoundTrip
    self.tokenMajorTrace = tokenMajorTrace
    self.headMajorTrace = headMajorTrace
  }
}

public typealias KVLayoutShootoutImplementation = (
  _ logicalValues: FloatTensor,
  _ configuration: KVCacheConfiguration,
  _ layer: Int,
  _ head: Int
) throws -> KVLayoutShootoutResult

public struct KVLayoutBenchmarkReport: Sendable, Equatable {
  public let configuration: KVCacheConfiguration
  public let iterations: Int
  public let tokenMajorNanoseconds: UInt64
  public let headMajorNanoseconds: UInt64
  public let checksum: Float

  public init(
    configuration: KVCacheConfiguration,
    iterations: Int,
    tokenMajorNanoseconds: UInt64,
    headMajorNanoseconds: UInt64,
    checksum: Float
  ) {
    self.configuration = configuration
    self.iterations = iterations
    self.tokenMajorNanoseconds = tokenMajorNanoseconds
    self.headMajorNanoseconds = headMajorNanoseconds
    self.checksum = checksum
  }

  public func rendered() -> String {
    """
    KV layout CPU read benchmark (batch=1, L=\(configuration.layerCount), T=\(configuration.capacity), Hkv=\(configuration.keyValueHeadCount), D=\(configuration.headDimension), iterations=\(iterations))
    token-major [L,T,H,D]: \(tokenMajorNanoseconds) ns
    head-major  [L,H,T,D]: \(headMajorNanoseconds) ns
    checksum: \(checksum)
    This outcome is specific to this shape, machine, build configuration, and access loop.
    """
  }
}

public enum P024KVLayoutShootoutJudge {
  public static func evaluate(_ implementation: KVLayoutShootoutImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let configuration = try KVCacheConfiguration(
        layerCount: 2, keyValueHeadCount: 2, headDimension: 3, capacity: 4)
      let logical = try FloatTensor(
        (0..<configuration.elementsPerTensor).map(Float.init), shape: [2, 4, 2, 3])
      let actual = try implementation(logical, configuration, 1, 1)
      if actual.tokenMajorRoundTrip == logical, actual.headMajorRoundTrip == logical {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "copy and round-trip both layouts",
          message: "at least one physical layout did not reconstruct [L,T,H,D] values"))
      }
      let expectedTokenOffsets = [27, 28, 29, 33, 34, 35, 39, 40, 41, 45, 46, 47]
      let expectedHeadOffsets = Array(36..<48)
      if actual.tokenMajorTrace.offsets == expectedTokenOffsets,
        actual.tokenMajorTrace.contiguousReadSpans == 4,
        actual.headMajorTrace.offsets == expectedHeadOffsets,
        actual.headMajorTrace.contiguousReadSpans == 1,
        actual.tokenMajorTrace.bytesRead == 48,
        actual.headMajorTrace.bytesRead == 48
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "deterministic head-read trace",
          message: "offsets, span count, or byte count does not match the two formulas"))
      }
      let token = KVLayoutDescriptor(kind: .tokenMajor, configuration: configuration)
      let head = KVLayoutDescriptor(kind: .headMajor, configuration: configuration)
      if try token.offset(layer: 1, slot: 2, head: 1, feature: 2) == 41,
        try head.offset(layer: 1, slot: 2, head: 1, feature: 2) == 44
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(caseName: "worked offsets", message: "offset formula mismatch"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "layout shootout", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 3, failures: failures)
  }
}
