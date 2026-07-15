import InferenceSchoolCore

public enum P045BatchSchedulingExercise {
  public static func simulate(
    _ request: SchedulingSimulationRequest
  ) throws -> BatchingSimulationReport {
    try P045BatchSchedulingContract.validate(request)
    // TODO: model static groups and continuous slot refill with explicit prefill events.
    return BatchingSimulationReport(
      policy: request.policy,
      timingUnitLabel: P045BatchSchedulingContract.timingUnitLabel,
      timeline: [],
      requests: [],
      makespan: 1,
      totalTokens: 0,
      throughputTokensPerUnit: 0,
      slotUtilization: 0)
  }
}