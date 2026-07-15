import InferenceSchoolCore

public enum P041BufferPlanningSolution {
  public static func plan(
    lifetimes: [BufferLifetime],
    strategy: ArenaAllocationStrategy
  ) throws -> ArenaPlan {
    try P041BufferPlanningContract.validate(lifetimes)
    let ordered = lifetimes.sorted {
      if $0.firstOperation != $1.firstOperation {
        return $0.firstOperation < $1.firstOperation
      }
      if $0.lastOperation != $1.lastOperation {
        return $0.lastOperation < $1.lastOperation
      }
      return $0.name < $1.name
    }
    var placements: [ArenaPlacement] = []
    var arenaByteCount = 0

    for lifetime in ordered {
      let livePlacements = placements.filter {
        $0.lastOperation >= lifetime.firstOperation
      }
      var liveRanges: [(Int, Int)] = livePlacements.map { placement in
        (placement.offset, placement.offset + placement.byteSize)
      }
      liveRanges.sort { lhs, rhs in
        lhs.0 == rhs.0 ? lhs.1 < rhs.1 : lhs.0 < rhs.0
      }
      let merged = merge(liveRanges)
      var candidates: [(offset: Int, available: Int)] = []
      var cursor = 0
      for range in merged {
        let aligned = try align(cursor, to: lifetime.alignment, name: lifetime.name)
        if aligned <= range.0, range.0 - aligned >= lifetime.byteSize {
          candidates.append((aligned, range.0 - aligned))
        }
        cursor = max(cursor, range.1)
      }
      let alignedCursor = try align(cursor, to: lifetime.alignment, name: lifetime.name)
      if alignedCursor <= arenaByteCount,
        arenaByteCount - alignedCursor >= lifetime.byteSize
      {
        candidates.append((alignedCursor, arenaByteCount - alignedCursor))
      }
      let offset: Int
      if candidates.isEmpty {
        offset = try align(arenaByteCount, to: lifetime.alignment, name: lifetime.name)
      } else {
        switch strategy {
        case .firstFit:
          offset = candidates.min { $0.offset < $1.offset }!.offset
        case .bestFit:
          offset = candidates.min {
            let leftRemainder = $0.available - lifetime.byteSize
            let rightRemainder = $1.available - lifetime.byteSize
            return leftRemainder == rightRemainder
              ? $0.offset < $1.offset
              : leftRemainder < rightRemainder
          }!.offset
        }
      }
      let (end, overflow) = offset.addingReportingOverflow(lifetime.byteSize)
      guard !overflow else {
        throw ArenaPlanningError.arithmeticOverflow(context: "placement of \(lifetime.name)")
      }
      placements.append(ArenaPlacement(lifetime: lifetime, offset: offset))
      arenaByteCount = max(arenaByteCount, end)
    }

    let naiveByteCount = try lifetimes.reduce(0) { partial, lifetime in
      let (next, overflow) = partial.addingReportingOverflow(lifetime.byteSize)
      guard !overflow else {
        throw ArenaPlanningError.arithmeticOverflow(context: "naive byte count")
      }
      return next
    }
    let operations = Set(lifetimes.flatMap { [$0.firstOperation, $0.lastOperation] })
    var peakLiveBytes = 0
    for operation in operations {
      let liveBytes = try lifetimes
        .filter { $0.firstOperation <= operation && operation <= $0.lastOperation }
        .reduce(0) { partial, lifetime in
          let (next, overflow) = partial.addingReportingOverflow(lifetime.byteSize)
          guard !overflow else {
            throw ArenaPlanningError.arithmeticOverflow(context: "peak live bytes")
          }
          return next
        }
      peakLiveBytes = max(peakLiveBytes, liveBytes)
    }
    let reuseAssignments = placements.enumerated().map { index, placement in
      let reused = placements[..<index].filter { prior in
        !prior.lifetime.overlapsLifetime(of: placement.lifetime)
          && P041BufferPlanningContract.rangesOverlap(
            prior.offset, prior.byteSize, placement.offset, placement.byteSize)
      }.map(\.name).sorted()
      return ArenaReuseAssignment(buffer: placement.name, reusesStorageFrom: reused)
    }
    let plan = ArenaPlan(
      strategy: strategy,
      placements: placements,
      arenaByteCount: arenaByteCount,
      peakLiveBytes: peakLiveBytes,
      naiveByteCount: naiveByteCount,
      reuseAssignments: reuseAssignments)
    try P041BufferPlanningContract.validate(plan: plan, for: lifetimes)
    return plan
  }

  public static func compareDecoderPlans(
    model: MiniDecoderModel,
    prefillTokenCount: Int,
    cachedTokenCount: Int,
    strategy: ArenaAllocationStrategy = .firstFit
  ) throws -> DecoderArenaComparison {
    DecoderArenaComparison(
      prefill: try plan(
        lifetimes: MiniDecoderBufferSchedules.prefill(
          model: model, tokenCount: prefillTokenCount),
        strategy: strategy),
      decode: try plan(
        lifetimes: MiniDecoderBufferSchedules.decode(
          model: model, cachedTokenCount: cachedTokenCount),
        strategy: strategy))
  }

  private static func merge(_ ranges: [(Int, Int)]) -> [(Int, Int)] {
    var merged: [(Int, Int)] = []
    for range in ranges {
      guard let last = merged.last, range.0 <= last.1 else {
        merged.append(range)
        continue
      }
      merged[merged.count - 1].1 = max(last.1, range.1)
    }
    return merged
  }

  private static func align(_ value: Int, to alignment: Int, name: String) throws -> Int {
    let mask = alignment - 1
    let (sum, overflow) = value.addingReportingOverflow(mask)
    guard !overflow else {
      throw ArenaPlanningError.arithmeticOverflow(context: "alignment of \(name)")
    }
    return sum & ~mask
  }
}