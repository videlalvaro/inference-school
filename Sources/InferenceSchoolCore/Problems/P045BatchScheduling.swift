import Foundation

public enum BatchingPolicy: String, Sendable, Equatable, Codable {
  case staticBatching
  case continuousBatching
}

public enum SchedulingEventKind: String, Sendable, Equatable, Codable {
  case idle
  case prefill
  case decodeIteration
}

public enum BatchSchedulingError: Error, Equatable, LocalizedError {
  case noRequests
  case emptyRequestID(index: Int)
  case duplicateRequestID(String)
  case invalidArrivalTime(requestID: String, value: Int)
  case emptyPrompt(requestID: String)
  case invalidDecodeLength(requestID: String, value: Int)
  case invalidSlotCount(Int)
  case invalidCost(name: String, value: Int)
  case invalidReport(String)

  public var errorDescription: String? {
    switch self {
    case .noRequests:
      "A batching simulation requires at least one request."
    case let .emptyRequestID(index):
      "Request at index \(index) has an empty ID."
    case let .duplicateRequestID(id):
      "Request ID \(id) appears more than once."
    case let .invalidArrivalTime(id, value):
      "Request \(id) has invalid arrival time \(value)."
    case let .emptyPrompt(id):
      "Request \(id) must contain at least one prompt token."
    case let .invalidDecodeLength(id, value):
      "Request \(id) has invalid decode length \(value)."
    case let .invalidSlotCount(value):
      "Slot count must be positive; received \(value)."
    case let .invalidCost(name, value):
      "Cost \(name) must be nonnegative and produce positive stage durations; received \(value)."
    case let .invalidReport(message):
      message
    }
  }
}

public struct ScheduledRequest: Sendable, Equatable {
  public let id: String
  public let arrivalTime: Int
  public let promptTokenIDs: [Int]
  public let decodeTokenCount: Int

  public init(
    id: String,
    arrivalTime: Int,
    promptTokenIDs: [Int],
    decodeTokenCount: Int
  ) {
    self.id = id
    self.arrivalTime = arrivalTime
    self.promptTokenIDs = promptTokenIDs
    self.decodeTokenCount = decodeTokenCount
  }
}

public struct BatchIterationCostModel: Sendable, Equatable {
  public let prefillFixedUnits: Int
  public let prefillUnitsPerPromptToken: Int
  public let decodeFixedUnits: Int
  public let decodeUnitsPerActiveRequest: Int

  public init(
    prefillFixedUnits: Int,
    prefillUnitsPerPromptToken: Int,
    decodeFixedUnits: Int,
    decodeUnitsPerActiveRequest: Int
  ) {
    self.prefillFixedUnits = prefillFixedUnits
    self.prefillUnitsPerPromptToken = prefillUnitsPerPromptToken
    self.decodeFixedUnits = decodeFixedUnits
    self.decodeUnitsPerActiveRequest = decodeUnitsPerActiveRequest
  }

  public func prefillCost(for requests: [ScheduledRequest]) -> Int {
    prefillFixedUnits
      + prefillUnitsPerPromptToken * (requests.map(\.promptTokenIDs.count).max() ?? 0)
  }

  public func decodeIterationCost(activeRequestCount: Int) -> Int {
    decodeFixedUnits + decodeUnitsPerActiveRequest * activeRequestCount
  }
}

public struct SchedulingSimulationRequest: Sendable, Equatable {
  public let requests: [ScheduledRequest]
  public let policy: BatchingPolicy
  public let slotCount: Int
  public let costModel: BatchIterationCostModel

  public init(
    requests: [ScheduledRequest],
    policy: BatchingPolicy,
    slotCount: Int,
    costModel: BatchIterationCostModel
  ) {
    self.requests = requests
    self.policy = policy
    self.slotCount = slotCount
    self.costModel = costModel
  }
}

public struct ScheduledToken: Sendable, Equatable {
  public let requestID: String
  public let tokenID: Int
  public let sequenceIndex: Int

  public init(requestID: String, tokenID: Int, sequenceIndex: Int) {
    self.requestID = requestID
    self.tokenID = tokenID
    self.sequenceIndex = sequenceIndex
  }
}

public struct SchedulingEvent: Sendable, Equatable {
  public let kind: SchedulingEventKind
  public let startTime: Int
  public let endTime: Int
  public let requestIDs: [String]
  public let occupiedSlotCount: Int
  public let generatedTokens: [ScheduledToken]

  public init(
    kind: SchedulingEventKind,
    startTime: Int,
    endTime: Int,
    requestIDs: [String],
    occupiedSlotCount: Int,
    generatedTokens: [ScheduledToken]
  ) {
    self.kind = kind
    self.startTime = startTime
    self.endTime = endTime
    self.requestIDs = requestIDs
    self.occupiedSlotCount = occupiedSlotCount
    self.generatedTokens = generatedTokens
  }
}

public struct RequestSchedulingMetrics: Sendable, Equatable {
  public let requestID: String
  public let arrivalTime: Int
  public let startTime: Int
  public let finishTime: Int
  public let latency: Int
  public let generatedTokenIDs: [Int]

  public init(
    requestID: String,
    arrivalTime: Int,
    startTime: Int,
    finishTime: Int,
    latency: Int,
    generatedTokenIDs: [Int]
  ) {
    self.requestID = requestID
    self.arrivalTime = arrivalTime
    self.startTime = startTime
    self.finishTime = finishTime
    self.latency = latency
    self.generatedTokenIDs = generatedTokenIDs
  }
}

public struct BatchingSimulationReport: Sendable, Equatable {
  public let policy: BatchingPolicy
  public let timingUnitLabel: String
  public let timeline: [SchedulingEvent]
  public let requests: [RequestSchedulingMetrics]
  public let makespan: Int
  public let totalTokens: Int
  public let throughputTokensPerUnit: Double
  public let slotUtilization: Double

  public init(
    policy: BatchingPolicy,
    timingUnitLabel: String,
    timeline: [SchedulingEvent],
    requests: [RequestSchedulingMetrics],
    makespan: Int,
    totalTokens: Int,
    throughputTokensPerUnit: Double,
    slotUtilization: Double
  ) {
    self.policy = policy
    self.timingUnitLabel = timingUnitLabel
    self.timeline = timeline
    self.requests = requests
    self.makespan = makespan
    self.totalTokens = totalTokens
    self.throughputTokensPerUnit = throughputTokensPerUnit
    self.slotUtilization = slotUtilization
  }
}

public typealias BatchSchedulingImplementation = (
  SchedulingSimulationRequest
) throws -> BatchingSimulationReport

public enum P045SemanticExecutor {
  public static func expectedTokens(for request: ScheduledRequest) -> [Int] {
    var generated: [Int] = []
    for step in 0..<request.decodeTokenCount {
      generated.append(nextToken(
        requestID: request.id,
        promptTokenIDs: request.promptTokenIDs,
        generatedTokenIDs: generated,
        step: step))
    }
    return generated
  }

  public static func nextToken(
    requestID: String,
    promptTokenIDs: [Int],
    generatedTokenIDs: [Int],
    step: Int
  ) -> Int {
    let salt = requestID.utf8.reduce(0) { ($0 * 31 + Int($1)) % 251 }
    let previous = generatedTokenIDs.last ?? promptTokenIDs.last!
    return (salt + previous * 17 + step * 13) % 256
  }
}

public enum P045BatchSchedulingContract {
  public static let timingUnitLabel = "modeled scheduler units"

  public static func validate(_ request: SchedulingSimulationRequest) throws {
    guard !request.requests.isEmpty else { throw BatchSchedulingError.noRequests }
    guard request.slotCount > 0 else {
      throw BatchSchedulingError.invalidSlotCount(request.slotCount)
    }
    var ids = Set<String>()
    for (index, item) in request.requests.enumerated() {
      guard !item.id.isEmpty else { throw BatchSchedulingError.emptyRequestID(index: index) }
      guard ids.insert(item.id).inserted else {
        throw BatchSchedulingError.duplicateRequestID(item.id)
      }
      guard item.arrivalTime >= 0 else {
        throw BatchSchedulingError.invalidArrivalTime(
          requestID: item.id, value: item.arrivalTime)
      }
      guard !item.promptTokenIDs.isEmpty else {
        throw BatchSchedulingError.emptyPrompt(requestID: item.id)
      }
      guard item.decodeTokenCount > 0 else {
        throw BatchSchedulingError.invalidDecodeLength(
          requestID: item.id, value: item.decodeTokenCount)
      }
    }
    let costs: [(String, Int)] = [
      ("prefillFixedUnits", request.costModel.prefillFixedUnits),
      ("prefillUnitsPerPromptToken", request.costModel.prefillUnitsPerPromptToken),
      ("decodeFixedUnits", request.costModel.decodeFixedUnits),
      ("decodeUnitsPerActiveRequest", request.costModel.decodeUnitsPerActiveRequest),
    ]
    for (name, value) in costs where value < 0 {
      throw BatchSchedulingError.invalidCost(name: name, value: value)
    }
    guard request.costModel.prefillCost(for: request.requests) > 0 else {
      throw BatchSchedulingError.invalidCost(name: "prefill", value: 0)
    }
    guard request.costModel.decodeIterationCost(activeRequestCount: 1) > 0 else {
      throw BatchSchedulingError.invalidCost(name: "decode", value: 0)
    }
  }

  public static func validate(
    _ report: BatchingSimulationReport,
    for request: SchedulingSimulationRequest
  ) throws {
    guard report.policy == request.policy else {
      throw BatchSchedulingError.invalidReport("Report policy does not match the requested policy.")
    }
    guard report.timingUnitLabel == timingUnitLabel else {
      throw BatchSchedulingError.invalidReport("Scheduler timing must be labeled as modeled units.")
    }
    guard report.requests.count == request.requests.count else {
      throw BatchSchedulingError.invalidReport("Report omitted one or more requests.")
    }
    let metricsByID = Dictionary(uniqueKeysWithValues: report.requests.map { ($0.requestID, $0) })
    for item in request.requests {
      guard let metrics = metricsByID[item.id] else {
        throw BatchSchedulingError.invalidReport("Missing metrics for request \(item.id).")
      }
      guard metrics.arrivalTime == item.arrivalTime,
        metrics.startTime >= item.arrivalTime,
        metrics.finishTime >= metrics.startTime,
        metrics.latency == metrics.finishTime - metrics.arrivalTime,
        metrics.generatedTokenIDs == P045SemanticExecutor.expectedTokens(for: item)
      else {
        throw BatchSchedulingError.invalidReport(
          "Request \(item.id) has invalid timing or cross-contaminated token state.")
      }
    }
    guard report.makespan == (report.requests.map(\.finishTime).max() ?? 0) else {
      throw BatchSchedulingError.invalidReport("Makespan must equal the last finish time.")
    }
    let totalTokens = request.requests.reduce(0) {
      $0 + $1.promptTokenIDs.count + $1.decodeTokenCount
    }
    guard report.totalTokens == totalTokens else {
      throw BatchSchedulingError.invalidReport("Total tokens must include prompt and decode tokens.")
    }
    let expectedThroughput = Double(totalTokens) / Double(report.makespan)
    guard abs(report.throughputTokensPerUnit - expectedThroughput) < 1e-12 else {
      throw BatchSchedulingError.invalidReport("Throughput does not match total tokens / makespan.")
    }
    guard report.slotUtilization >= 0, report.slotUtilization <= 1 else {
      throw BatchSchedulingError.invalidReport("Slot utilization must be in [0, 1].")
    }
    var previousEnd = 0
    for event in report.timeline {
      guard event.startTime == previousEnd,
        event.endTime > event.startTime,
        event.occupiedSlotCount >= 0,
        event.occupiedSlotCount <= request.slotCount
      else {
        throw BatchSchedulingError.invalidReport("Timeline events must be contiguous and bounded.")
      }
      previousEnd = event.endTime
    }
    guard previousEnd == report.makespan else {
      throw BatchSchedulingError.invalidReport("Timeline must end at makespan.")
    }
  }
}

public enum P045BatchSchedulingJudge {
  public static func evaluate(_ implementation: BatchSchedulingImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      let workload = comparisonWorkload()
      let staticRequest = SchedulingSimulationRequest(
        requests: workload,
        policy: .staticBatching,
        slotCount: 2,
        costModel: comparisonCostModel())
      let continuousRequest = SchedulingSimulationRequest(
        requests: workload,
        policy: .continuousBatching,
        slotCount: 2,
        costModel: comparisonCostModel())
      let staticReport = try implementation(staticRequest)
      let continuousReport = try implementation(continuousRequest)
      try P045BatchSchedulingContract.validate(staticReport, for: staticRequest)
      try P045BatchSchedulingContract.validate(continuousReport, for: continuousRequest)

      let staticFinish = Dictionary(uniqueKeysWithValues:
        staticReport.requests.map { ($0.requestID, $0.finishTime) })
      if staticReport.makespan == 16,
        staticFinish == ["A": 6, "B": 2, "C": 16]
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "static group runs to completion",
          message: "expected makespan 16 and finish times A=6, B=2, C=16"))
      }

      let continuousFinish = Dictionary(uniqueKeysWithValues:
        continuousReport.requests.map { ($0.requestID, $0.finishTime) })
      if continuousReport.makespan == 12,
        continuousFinish == ["A": 11, "B": 2, "C": 12]
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "continuous refill with explicit prefill stall",
          message: "expected makespan 12 and finish times A=11, B=2, C=12"))
      }

      let staticA = staticReport.requests.first { $0.requestID == "A" }!
      let continuousA = continuousReport.requests.first { $0.requestID == "A" }!
      if continuousReport.throughputTokensPerUnit > staticReport.throughputTokensPerUnit,
        continuousA.latency > staticA.latency
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "throughput improves while one latency worsens",
          message: "the fixture must demonstrate the policy tradeoff, not universal dominance"))
      }

      if staticReport.requests.allSatisfy({ metrics in
        guard let item = workload.first(where: { $0.id == metrics.requestID }) else { return false }
        return metrics.generatedTokenIDs == P045SemanticExecutor.expectedTokens(for: item)
      }), continuousReport.requests.allSatisfy({ metrics in
        guard let item = workload.first(where: { $0.id == metrics.requestID }) else { return false }
        return metrics.generatedTokenIDs == P045SemanticExecutor.expectedTokens(for: item)
      }) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "request sequence state remains isolated",
          message: "generated tokens crossed request boundaries"))
      }

      passed += expectError(name: "reject duplicate IDs", failures: &failures) {
        _ = try implementation(SchedulingSimulationRequest(
          requests: [workload[0], workload[0]],
          policy: .staticBatching,
          slotCount: 2,
          costModel: comparisonCostModel()))
      }
      passed += expectError(name: "reject zero slots", failures: &failures) {
        _ = try implementation(SchedulingSimulationRequest(
          requests: workload,
          policy: .continuousBatching,
          slotCount: 0,
          costModel: comparisonCostModel()))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 6, failures: failures)
  }

  public static func comparisonWorkload() -> [ScheduledRequest] {
    [
      ScheduledRequest(id: "A", arrivalTime: 0, promptTokenIDs: [10], decodeTokenCount: 5),
      ScheduledRequest(id: "B", arrivalTime: 0, promptTokenIDs: [20], decodeTokenCount: 1),
      ScheduledRequest(id: "C", arrivalTime: 1, promptTokenIDs: [30, 31, 32, 33, 34], decodeTokenCount: 5),
    ]
  }

  public static func comparisonCostModel() -> BatchIterationCostModel {
    BatchIterationCostModel(
      prefillFixedUnits: 0,
      prefillUnitsPerPromptToken: 1,
      decodeFixedUnits: 1,
      decodeUnitsPerActiveRequest: 0)
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