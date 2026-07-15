import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P041BufferPlanningTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P041BufferPlanningJudge.evaluate(P041BufferPlanningSolution.plan)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testPlanValidationRejectsSimultaneousPhysicalOverlap() throws {
    let lifetimes = [
      BufferLifetime(name: "a", firstOperation: 0, lastOperation: 1, byteSize: 8, alignment: 8),
      BufferLifetime(name: "b", firstOperation: 1, lastOperation: 2, byteSize: 8, alignment: 8),
    ]
    let invalid = ArenaPlan(
      strategy: .firstFit,
      placements: lifetimes.map { ArenaPlacement(lifetime: $0, offset: 0) },
      arenaByteCount: 8,
      peakLiveBytes: 16,
      naiveByteCount: 16,
      reuseAssignments: [])
    XCTAssertThrowsError(try P041BufferPlanningContract.validate(plan: invalid, for: lifetimes))
  }

  func testDecoderPlansAreExecutableAndDecodeIsSmaller() throws {
    let model = try EducationalMiniModelFixture.make()
    let comparison = try P041BufferPlanningSolution.compareDecoderPlans(
      model: model,
      prefillTokenCount: 16,
      cachedTokenCount: 16)
    XCTAssertLessThan(comparison.prefill.arenaByteCount, comparison.prefill.naiveByteCount)
    XCTAssertLessThan(comparison.decode.arenaByteCount, comparison.prefill.arenaByteCount)
    XCTAssertFalse(comparison.prefill.reuseAssignments.allSatisfy(\.reusesStorageFrom.isEmpty))
  }

  func testBestFitIsDeterministicAcrossInputOrdering() throws {
    let lifetimes = try MiniDecoderBufferSchedules.prefill(
      model: EducationalMiniModelFixture.make(), tokenCount: 4)
    XCTAssertEqual(
      try P041BufferPlanningSolution.plan(lifetimes: lifetimes, strategy: .bestFit),
      try P041BufferPlanningSolution.plan(
        lifetimes: Array(lifetimes.reversed()), strategy: .bestFit))
  }
}