import InferenceSchoolCore

public enum P041BufferPlanningExercise {
  public static func plan(
    lifetimes: [BufferLifetime],
    strategy: ArenaAllocationStrategy
  ) throws -> ArenaPlan {
    try P041BufferPlanningContract.validate(lifetimes)
    var offset = 0
    var placements: [ArenaPlacement] = []
    for lifetime in lifetimes.sorted(by: { $0.name < $1.name }) {
      let remainder = offset % lifetime.alignment
      if remainder != 0 { offset += lifetime.alignment - remainder }
      placements.append(ArenaPlacement(lifetime: lifetime, offset: offset))
      offset += lifetime.byteSize
    }

    // TODO: release ranges when lifetimes end and choose a deterministic aligned
    // first-fit or best-fit gap instead of allocating every buffer sequentially.
    return ArenaPlan(
      strategy: strategy,
      placements: placements,
      arenaByteCount: offset,
      peakLiveBytes: lifetimes.map(\.byteSize).reduce(0, +),
      naiveByteCount: lifetimes.map(\.byteSize).reduce(0, +),
      reuseAssignments: placements.map {
        ArenaReuseAssignment(buffer: $0.name, reusesStorageFrom: [])
      })
  }
}