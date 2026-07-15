import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P045BatchSchedulingTests: XCTestCase {
  func testCanonicalSchedulerPassesJudge() {
    let report = P045BatchSchedulingJudge.evaluate(P045BatchSchedulingSolution.simulate)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testContinuousBatchingImprovesFixtureThroughputButWorsensALatency() throws {
    let workload = P045BatchSchedulingJudge.comparisonWorkload()
    let cost = P045BatchSchedulingJudge.comparisonCostModel()
    let staticReport = try P045BatchSchedulingSolution.simulate(SchedulingSimulationRequest(
      requests: workload,
      policy: .staticBatching,
      slotCount: 2,
      costModel: cost))
    let continuousReport = try P045BatchSchedulingSolution.simulate(SchedulingSimulationRequest(
      requests: workload,
      policy: .continuousBatching,
      slotCount: 2,
      costModel: cost))
    XCTAssertEqual(staticReport.makespan, 16)
    XCTAssertEqual(continuousReport.makespan, 12)
    XCTAssertGreaterThan(
      continuousReport.throughputTokensPerUnit,
      staticReport.throughputTokensPerUnit)
    XCTAssertGreaterThan(
      continuousReport.requests.first { $0.requestID == "A" }!.latency,
      staticReport.requests.first { $0.requestID == "A" }!.latency)
  }

  func testTimelinePreservesPerRequestTokenOrderAndState() throws {
    let workload = P045BatchSchedulingJudge.comparisonWorkload()
    let report = try P045BatchSchedulingSolution.simulate(SchedulingSimulationRequest(
      requests: workload,
      policy: .continuousBatching,
      slotCount: 2,
      costModel: P045BatchSchedulingJudge.comparisonCostModel()))
    for item in workload {
      let metrics = report.requests.first { $0.requestID == item.id }!
      XCTAssertEqual(metrics.generatedTokenIDs, P045SemanticExecutor.expectedTokens(for: item))
      let timelineTokens = report.timeline.flatMap(\.generatedTokens)
        .filter { $0.requestID == item.id }
        .sorted { $0.sequenceIndex < $1.sequenceIndex }
        .map(\.tokenID)
      XCTAssertEqual(timelineTokens, metrics.generatedTokenIDs)
    }
  }

  func testJudgeRejectsCrossContaminatedOrMissingRequests() {
    let report = P045BatchSchedulingJudge.evaluate { request in
      try P045BatchSchedulingContract.validate(request)
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
    XCTAssertFalse(report.isPassing)
  }
}