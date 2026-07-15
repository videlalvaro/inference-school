import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P044PrefillDecodeProfilingTests: XCTestCase {
  func testCanonicalProfilerPassesJudge() {
    let report = P044ProfilingJudge.evaluate(P044PrefillDecodeProfilingSolution.profile)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testStatisticsAreDeterministicAndUseNearestRankPercentile() throws {
    let statistics = try P044LatencyStatistics.summarize(
      [9, 1, 5, 7, 3], percentile: 0.8)
    XCTAssertEqual(statistics.medianNanoseconds, 5)
    XCTAssertEqual(statistics.percentileNanoseconds, 7)
    XCTAssertEqual(statistics.minimumNanoseconds, 1)
    XCTAssertEqual(statistics.maximumNanoseconds, 9)
  }

  func testProfilerSeparatesPrefillAndDecodeSamples() throws {
    let request = PrefillDecodeProfilingRequest(
      model: try EducationalMiniModelFixture.make(layerCount: 1),
      promptTokenIDs: [1, 4, 2],
      decodeContextLengths: [3, 5],
      warmupIterations: 0,
      measuredTrials: 2,
      decodeStepsPerTrial: 2)
    let report = try P044PrefillDecodeProfilingSolution.profile(request)
    XCTAssertEqual(report.prefill.latency.samplesNanoseconds.count, 2)
    XCTAssertEqual(report.decode.map(\.initialContextLength), [3, 5])
    XCTAssertTrue(report.decode.allSatisfy { $0.perTokenLatency.samplesNanoseconds.count == 4 })
    XCTAssertTrue(report.decode.allSatisfy { $0.averageWorkPerToken.floatingPointOperations > 0 })
    XCTAssertEqual(report.backend, "CPU reference backend")
  }

  func testJudgeRejectsBlendedOrZeroTimingReport() {
    let report = P044ProfilingJudge.evaluate { request in
      let statistics = LatencyStatistics(
        samplesNanoseconds: Array(repeating: 0, count: request.measuredTrials),
        medianNanoseconds: 0,
        percentile: request.percentile,
        percentileNanoseconds: 0,
        minimumNanoseconds: 0,
        maximumNanoseconds: 0)
      return PrefillDecodeProfileReport(
        backend: "blended",
        clock: "none",
        timingBoundary: "none",
        warmupIterations: request.warmupIterations,
        measuredTrials: request.measuredTrials,
        prefill: PrefillProfile(
          stageName: "tokens/s",
          promptTokenCount: request.promptTokenIDs.count,
          latency: statistics,
          promptTokensPerSecond: 0,
          work: ProfilingWorkEstimate(
            floatingPointOperations: 0, estimatedWeightBytesRead: 0, cacheBytesWritten: 0)),
        decode: [])
    }
    XCTAssertFalse(report.isPassing)
  }
}