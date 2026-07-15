import InferenceSchoolCore

public enum P045BatchSchedulingSolution {
  public static func simulate(
    _ request: SchedulingSimulationRequest
  ) throws -> BatchingSimulationReport {
    try P045BatchSchedulingContract.validate(request)
    switch request.policy {
    case .staticBatching:
      return try simulateStatic(request)
    case .continuousBatching:
      return try simulateContinuous(request)
    }
  }

  private struct RuntimeRequest {
    let request: ScheduledRequest
    var startTime: Int?
    var finishTime: Int?
    var generatedTokenIDs: [Int] = []

    var remainingDecodeTokens: Int {
      request.decodeTokenCount - generatedTokenIDs.count
    }

    mutating func advance() -> ScheduledToken {
      let step = generatedTokenIDs.count
      let token = P045SemanticExecutor.nextToken(
        requestID: request.id,
        promptTokenIDs: request.promptTokenIDs,
        generatedTokenIDs: generatedTokenIDs,
        step: step)
      generatedTokenIDs.append(token)
      return ScheduledToken(requestID: request.id, tokenID: token, sequenceIndex: step)
    }
  }

  private struct SimulationState {
    var time = 0
    var timeline: [SchedulingEvent] = []
    var occupiedSlotUnits = 0
    var runtimes: [String: RuntimeRequest]
    var pendingIDs: [String]

    init(requests: [ScheduledRequest]) {
      runtimes = Dictionary(uniqueKeysWithValues: requests.map {
        ($0.id, RuntimeRequest(request: $0))
      })
      pendingIDs = requests.enumerated().sorted {
        if $0.element.arrivalTime == $1.element.arrivalTime {
          return $0.offset < $1.offset
        }
        return $0.element.arrivalTime < $1.element.arrivalTime
      }.map(\.element.id)
    }

    mutating func appendEvent(
      kind: SchedulingEventKind,
      duration: Int,
      requestIDs: [String],
      occupiedSlotCount: Int,
      generatedTokens: [ScheduledToken] = []
    ) {
      timeline.append(SchedulingEvent(
        kind: kind,
        startTime: time,
        endTime: time + duration,
        requestIDs: requestIDs,
        occupiedSlotCount: occupiedSlotCount,
        generatedTokens: generatedTokens))
      occupiedSlotUnits += duration * occupiedSlotCount
      time += duration
    }

    mutating func advanceToNextArrival() {
      guard let id = pendingIDs.first, let next = runtimes[id]?.request.arrivalTime,
        next > time
      else { return }
      appendEvent(kind: .idle, duration: next - time, requestIDs: [], occupiedSlotCount: 0)
    }

    mutating func admit(upTo count: Int) -> [String] {
      var admitted: [String] = []
      while admitted.count < count,
        let id = pendingIDs.first,
        let runtime = runtimes[id],
        runtime.request.arrivalTime <= time
      {
        pendingIDs.removeFirst()
        var updated = runtime
        updated.startTime = time
        runtimes[id] = updated
        admitted.append(id)
      }
      return admitted
    }

    func report(for request: SchedulingSimulationRequest) throws -> BatchingSimulationReport {
      let metrics = request.requests.map { item -> RequestSchedulingMetrics in
        let runtime = runtimes[item.id]!
        return RequestSchedulingMetrics(
          requestID: item.id,
          arrivalTime: item.arrivalTime,
          startTime: runtime.startTime!,
          finishTime: runtime.finishTime!,
          latency: runtime.finishTime! - item.arrivalTime,
          generatedTokenIDs: runtime.generatedTokenIDs)
      }
      let totalTokens = request.requests.reduce(0) {
        $0 + $1.promptTokenIDs.count + $1.decodeTokenCount
      }
      let result = BatchingSimulationReport(
        policy: request.policy,
        timingUnitLabel: P045BatchSchedulingContract.timingUnitLabel,
        timeline: timeline,
        requests: metrics,
        makespan: time,
        totalTokens: totalTokens,
        throughputTokensPerUnit: Double(totalTokens) / Double(time),
        slotUtilization: Double(occupiedSlotUnits) / Double(request.slotCount * time))
      try P045BatchSchedulingContract.validate(result, for: request)
      return result
    }
  }

  private static func simulateStatic(
    _ request: SchedulingSimulationRequest
  ) throws -> BatchingSimulationReport {
    var state = SimulationState(requests: request.requests)
    while !state.pendingIDs.isEmpty {
      state.advanceToNextArrival()
      var active = state.admit(upTo: request.slotCount)
      let admittedRequests = active.map { state.runtimes[$0]!.request }
      state.appendEvent(
        kind: .prefill,
        duration: request.costModel.prefillCost(for: admittedRequests),
        requestIDs: active,
        occupiedSlotCount: active.count)
      while !active.isEmpty {
        var tokens: [ScheduledToken] = []
        for id in active {
          var runtime = state.runtimes[id]!
          tokens.append(runtime.advance())
          state.runtimes[id] = runtime
        }
        let participants = active
        state.appendEvent(
          kind: .decodeIteration,
          duration: request.costModel.decodeIterationCost(activeRequestCount: participants.count),
          requestIDs: participants,
          occupiedSlotCount: participants.count,
          generatedTokens: tokens)
        active.removeAll { id in
          var runtime = state.runtimes[id]!
          guard runtime.remainingDecodeTokens == 0 else { return false }
          runtime.finishTime = state.time
          state.runtimes[id] = runtime
          return true
        }
      }
    }
    return try state.report(for: request)
  }

  private static func simulateContinuous(
    _ request: SchedulingSimulationRequest
  ) throws -> BatchingSimulationReport {
    var state = SimulationState(requests: request.requests)
    var active: [String] = []
    while !state.pendingIDs.isEmpty || !active.isEmpty {
      if active.isEmpty {
        state.advanceToNextArrival()
      }
      let admitted = state.admit(upTo: request.slotCount - active.count)
      if !admitted.isEmpty {
        active.append(contentsOf: admitted)
        let admittedRequests = admitted.map { state.runtimes[$0]!.request }
        state.appendEvent(
          kind: .prefill,
          duration: request.costModel.prefillCost(for: admittedRequests),
          requestIDs: admitted,
          occupiedSlotCount: active.count)
      }
      guard !active.isEmpty else { continue }

      var tokens: [ScheduledToken] = []
      for id in active {
        var runtime = state.runtimes[id]!
        tokens.append(runtime.advance())
        state.runtimes[id] = runtime
      }
      let participants = active
      state.appendEvent(
        kind: .decodeIteration,
        duration: request.costModel.decodeIterationCost(activeRequestCount: participants.count),
        requestIDs: participants,
        occupiedSlotCount: participants.count,
        generatedTokens: tokens)
      active.removeAll { id in
        var runtime = state.runtimes[id]!
        guard runtime.remainingDecodeTokens == 0 else { return false }
        runtime.finishTime = state.time
        state.runtimes[id] = runtime
        return true
      }
    }
    return try state.report(for: request)
  }
}