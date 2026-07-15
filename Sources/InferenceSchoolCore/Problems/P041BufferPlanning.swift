import Foundation

public enum ArenaPlanningError: Error, Equatable, LocalizedError {
  case emptyName(index: Int)
  case duplicateName(String)
  case invalidOperationRange(name: String, first: Int, last: Int)
  case invalidByteSize(name: String, value: Int)
  case invalidAlignment(name: String, value: Int)
  case arithmeticOverflow(context: String)
  case invalidPlacement(name: String)
  case simultaneousLiveOverlap(first: String, second: String)

  public var errorDescription: String? {
    switch self {
    case .emptyName(let index):
      "Buffer lifetime \(index) must have a nonempty name."
    case .duplicateName(let name):
      "Buffer lifetime name '\(name)' appears more than once."
    case .invalidOperationRange(let name, let first, let last):
      "Buffer '\(name)' requires 0 <= firstOperation <= lastOperation; received \(first)...\(last)."
    case .invalidByteSize(let name, let value):
      "Buffer '\(name)' byte size must be positive; received \(value)."
    case .invalidAlignment(let name, let value):
      "Buffer '\(name)' alignment must be a positive power of two; received \(value)."
    case .arithmeticOverflow(let context):
      "Integer arithmetic overflowed while computing \(context)."
    case .invalidPlacement(let name):
      "Arena placement for '\(name)' is out of bounds, misaligned, or has the wrong size."
    case .simultaneousLiveOverlap(let first, let second):
      "Simultaneously live buffers '\(first)' and '\(second)' overlap in the arena."
    }
  }
}

public enum ArenaAllocationStrategy: String, Sendable, Equatable, Codable {
  case firstFit
  case bestFit
}

public struct BufferLifetime: Sendable, Equatable {
  public let name: String
  public let firstOperation: Int
  public let lastOperation: Int
  public let byteSize: Int
  public let alignment: Int

  public init(
    name: String,
    firstOperation: Int,
    lastOperation: Int,
    byteSize: Int,
    alignment: Int
  ) {
    self.name = name
    self.firstOperation = firstOperation
    self.lastOperation = lastOperation
    self.byteSize = byteSize
    self.alignment = alignment
  }

  public func overlapsLifetime(of other: BufferLifetime) -> Bool {
    firstOperation <= other.lastOperation && other.firstOperation <= lastOperation
  }
}

public struct ArenaPlacement: Sendable, Equatable {
  public let name: String
  public let offset: Int
  public let byteSize: Int
  public let alignment: Int
  public let firstOperation: Int
  public let lastOperation: Int

  public init(lifetime: BufferLifetime, offset: Int) {
    name = lifetime.name
    self.offset = offset
    byteSize = lifetime.byteSize
    alignment = lifetime.alignment
    firstOperation = lifetime.firstOperation
    lastOperation = lifetime.lastOperation
  }

  public var lifetime: BufferLifetime {
    BufferLifetime(
      name: name,
      firstOperation: firstOperation,
      lastOperation: lastOperation,
      byteSize: byteSize,
      alignment: alignment)
  }
}

public struct ArenaReuseAssignment: Sendable, Equatable {
  public let buffer: String
  public let reusesStorageFrom: [String]

  public init(buffer: String, reusesStorageFrom: [String]) {
    self.buffer = buffer
    self.reusesStorageFrom = reusesStorageFrom
  }
}

public struct ArenaPlan: Sendable, Equatable {
  public let strategy: ArenaAllocationStrategy
  public let placements: [ArenaPlacement]
  public let arenaByteCount: Int
  public let peakLiveBytes: Int
  public let naiveByteCount: Int
  public let reuseAssignments: [ArenaReuseAssignment]

  public init(
    strategy: ArenaAllocationStrategy,
    placements: [ArenaPlacement],
    arenaByteCount: Int,
    peakLiveBytes: Int,
    naiveByteCount: Int,
    reuseAssignments: [ArenaReuseAssignment]
  ) {
    self.strategy = strategy
    self.placements = placements
    self.arenaByteCount = arenaByteCount
    self.peakLiveBytes = peakLiveBytes
    self.naiveByteCount = naiveByteCount
    self.reuseAssignments = reuseAssignments
  }
}

public struct DecoderArenaComparison: Sendable, Equatable {
  public let prefill: ArenaPlan
  public let decode: ArenaPlan

  public init(prefill: ArenaPlan, decode: ArenaPlan) {
    self.prefill = prefill
    self.decode = decode
  }
}

public typealias ArenaPlanningImplementation = (
  _ lifetimes: [BufferLifetime],
  _ strategy: ArenaAllocationStrategy
) throws -> ArenaPlan

public enum P041BufferPlanningContract {
  public static func validate(_ lifetimes: [BufferLifetime]) throws {
    var names: Set<String> = []
    for (index, lifetime) in lifetimes.enumerated() {
      guard !lifetime.name.isEmpty else { throw ArenaPlanningError.emptyName(index: index) }
      guard names.insert(lifetime.name).inserted else {
        throw ArenaPlanningError.duplicateName(lifetime.name)
      }
      guard lifetime.firstOperation >= 0,
        lifetime.lastOperation >= lifetime.firstOperation
      else {
        throw ArenaPlanningError.invalidOperationRange(
          name: lifetime.name,
          first: lifetime.firstOperation,
          last: lifetime.lastOperation)
      }
      guard lifetime.byteSize > 0 else {
        throw ArenaPlanningError.invalidByteSize(
          name: lifetime.name, value: lifetime.byteSize)
      }
      guard lifetime.alignment > 0,
        lifetime.alignment.isMultiple(of: 2) || lifetime.alignment == 1,
        lifetime.alignment.nonzeroBitCount == 1
      else {
        throw ArenaPlanningError.invalidAlignment(
          name: lifetime.name, value: lifetime.alignment)
      }
      let (_, overflow) = lifetime.byteSize.addingReportingOverflow(
        lifetime.alignment - 1)
      guard !overflow else {
        throw ArenaPlanningError.arithmeticOverflow(context: "aligned size of \(lifetime.name)")
      }
    }
  }

  public static func validate(
    plan: ArenaPlan,
    for lifetimes: [BufferLifetime]
  ) throws {
    try validate(lifetimes)
    guard plan.placements.count == lifetimes.count else {
      throw ArenaPlanningError.invalidPlacement(name: "placement count")
    }
    let byName = Dictionary(uniqueKeysWithValues: lifetimes.map { ($0.name, $0) })
    for placement in plan.placements {
      guard let lifetime = byName[placement.name], placement.lifetime == lifetime,
        placement.offset >= 0,
        placement.offset.isMultiple(of: placement.alignment)
      else { throw ArenaPlanningError.invalidPlacement(name: placement.name) }
      let (end, overflow) = placement.offset.addingReportingOverflow(placement.byteSize)
      guard !overflow, end <= plan.arenaByteCount else {
        throw ArenaPlanningError.invalidPlacement(name: placement.name)
      }
    }
    for firstIndex in plan.placements.indices {
      for secondIndex in plan.placements.indices where secondIndex > firstIndex {
        let first = plan.placements[firstIndex]
        let second = plan.placements[secondIndex]
        if first.lifetime.overlapsLifetime(of: second.lifetime),
          rangesOverlap(
            first.offset, first.byteSize,
            second.offset, second.byteSize)
        {
          throw ArenaPlanningError.simultaneousLiveOverlap(
            first: first.name, second: second.name)
        }
      }
    }
  }

  public static func rangesOverlap(
    _ firstOffset: Int,
    _ firstSize: Int,
    _ secondOffset: Int,
    _ secondSize: Int
  ) -> Bool {
    firstOffset < secondOffset + secondSize && secondOffset < firstOffset + firstSize
  }
}

public enum MiniDecoderBufferSchedules {
  public static func prefill(
    model: MiniDecoderModel,
    tokenCount: Int
  ) throws -> [BufferLifetime] {
    guard tokenCount > 0 else { throw MiniDecoderError.emptyPrompt }
    return try schedule(model: model, tokenCount: tokenCount, decode: false)
  }

  public static func decode(
    model: MiniDecoderModel,
    cachedTokenCount: Int
  ) throws -> [BufferLifetime] {
    guard cachedTokenCount > 0 else { throw KVCacheError.invalidTokenCount(cachedTokenCount) }
    return try schedule(model: model, tokenCount: 1, decode: true)
  }

  private static func schedule(
    model: MiniDecoderModel,
    tokenCount: Int,
    decode: Bool
  ) throws -> [BufferLifetime] {
    let configuration = model.configuration
    let floatBytes = MemoryLayout<Float>.stride
    func bytes(_ dimensions: [Int], _ name: String) throws -> Int {
      var count = 1
      for dimension in dimensions {
        let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
        guard !overflow else {
          throw ArenaPlanningError.arithmeticOverflow(context: "element count for \(name)")
        }
        count = next
      }
      let (result, overflow) = count.multipliedReportingOverflow(by: floatBytes)
      guard !overflow else {
        throw ArenaPlanningError.arithmeticOverflow(context: "byte count for \(name)")
      }
      return result
    }

    let prefix = decode ? "decode" : "prefill"
    let modelBytes = try bytes([tokenCount, configuration.modelDimension], "model state")
    let queryBytes = try bytes(
      [tokenCount, configuration.queryHeadCount, configuration.headDimension], "queries")
    let keyValueBytes = try bytes(
      [tokenCount, configuration.keyValueHeadCount, configuration.headDimension], "keys")
    let hiddenBytes = try bytes(
      [tokenCount, configuration.hiddenDimension], "MLP hidden")
    return [
      BufferLifetime(name: "\(prefix).residual", firstOperation: 0, lastOperation: 14, byteSize: modelBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).attention_norm", firstOperation: 1, lastOperation: 4, byteSize: modelBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).query", firstOperation: 2, lastOperation: 6, byteSize: queryBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).key", firstOperation: 3, lastOperation: 6, byteSize: keyValueBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).value", firstOperation: 4, lastOperation: 6, byteSize: keyValueBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).attention_output", firstOperation: 6, lastOperation: 8, byteSize: modelBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).post_attention", firstOperation: 8, lastOperation: 14, byteSize: modelBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).mlp_norm", firstOperation: 9, lastOperation: 12, byteSize: modelBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).gate", firstOperation: 10, lastOperation: 12, byteSize: hiddenBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).up", firstOperation: 11, lastOperation: 12, byteSize: hiddenBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).gated", firstOperation: 12, lastOperation: 13, byteSize: hiddenBytes, alignment: 64),
      BufferLifetime(name: "\(prefix).down", firstOperation: 13, lastOperation: 14, byteSize: modelBytes, alignment: 64),
    ]
  }
}

public enum P041BufferPlanningJudge {
  public static func evaluate(_ implementation: ArenaPlanningImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    let fixture = [
      BufferLifetime(name: "a", firstOperation: 0, lastOperation: 2, byteSize: 24, alignment: 8),
      BufferLifetime(name: "b", firstOperation: 1, lastOperation: 1, byteSize: 8, alignment: 16),
      BufferLifetime(name: "c", firstOperation: 2, lastOperation: 4, byteSize: 16, alignment: 8),
      BufferLifetime(name: "d", firstOperation: 3, lastOperation: 3, byteSize: 20, alignment: 4),
    ]
    do {
      let firstFit = try implementation(fixture, .firstFit)
      try P041BufferPlanningContract.validate(plan: firstFit, for: fixture)
      let offsets = Dictionary(uniqueKeysWithValues: firstFit.placements.map { ($0.name, $0.offset) })
      if offsets == ["a": 0, "b": 32, "c": 24, "d": 0],
        firstFit.arenaByteCount == 40,
        firstFit.peakLiveBytes == 40,
        firstFit.naiveByteCount == 68,
        firstFit.reuseAssignments.contains(where: {
          $0.buffer == "d" && $0.reusesStorageFrom.contains("a")
        })
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "deterministic aligned first-fit reuse",
          message: "expected offsets a=0, b=32, c=24, d=0 with a 40-byte arena and correct liveness statistics"))
      }

      let bestFit = try implementation(Array(fixture.reversed()), .bestFit)
      try P041BufferPlanningContract.validate(plan: bestFit, for: fixture)
      let repeated = try implementation(fixture, .bestFit)
      if bestFit == repeated, bestFit.strategy == .bestFit {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "best-fit is explicit and input-order deterministic",
          message: "sorting and tie-breaking must produce the same plan for the same named lifetimes"))
      }

      let overlap = [
        BufferLifetime(name: "live-a", firstOperation: 0, lastOperation: 2, byteSize: 8, alignment: 8),
        BufferLifetime(name: "live-b", firstOperation: 1, lastOperation: 1, byteSize: 8, alignment: 8),
      ]
      let overlapPlan = try implementation(overlap, .firstFit)
      try P041BufferPlanningContract.validate(plan: overlapPlan, for: overlap)
      if overlapPlan.placements[0].offset != overlapPlan.placements[1].offset {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "simultaneously live buffers never reuse",
          message: "overlapping lifetimes were assigned overlapping storage"))
      }

      let model = try EducationalMiniModelFixture.make()
      let prefill = try implementation(
        MiniDecoderBufferSchedules.prefill(model: model, tokenCount: 16), .firstFit)
      let decode = try implementation(
        MiniDecoderBufferSchedules.decode(model: model, cachedTokenCount: 16), .firstFit)
      if prefill.arenaByteCount < prefill.naiveByteCount,
        prefill.reuseAssignments.contains(where: { !$0.reusesStorageFrom.isEmpty }),
        decode.reuseAssignments.contains(where: { !$0.reusesStorageFrom.isEmpty }),
        decode.arenaByteCount < prefill.arenaByteCount
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "decoder-derived prefill and decode plans",
          message: "both schedules should reuse physical ranges and the one-token decode arena should be smaller"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "valid planning fixtures", message: error.localizedDescription))
    }

    passed += expectError(name: "reject duplicate names", failures: &failures) {
      _ = try implementation([
        BufferLifetime(name: "x", firstOperation: 0, lastOperation: 0, byteSize: 4, alignment: 4),
        BufferLifetime(name: "x", firstOperation: 1, lastOperation: 1, byteSize: 4, alignment: 4),
      ], .firstFit)
    }
    passed += expectError(name: "reject non-power-of-two alignment", failures: &failures) {
      _ = try implementation([
        BufferLifetime(name: "x", firstOperation: 0, lastOperation: 0, byteSize: 4, alignment: 3)
      ], .firstFit)
    }
    passed += expectError(name: "reject invalid operation range", failures: &failures) {
      _ = try implementation([
        BufferLifetime(name: "x", firstOperation: 2, lastOperation: 1, byteSize: 4, alignment: 4)
      ], .firstFit)
    }
    passed += expectError(name: "reject aligned-size overflow", failures: &failures) {
      _ = try implementation([
        BufferLifetime(
          name: "x", firstOperation: 0, lastOperation: 0, byteSize: Int.max, alignment: 8)
      ], .firstFit)
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 8, failures: failures)
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(caseName: name, message: "expected an error"))
      return 0
    } catch {
      return 1
    }
  }
}