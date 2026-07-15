import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P047CapstoneTests: XCTestCase {
  func testCanonicalCapstonePassesJudge() {
    let report = P047CapstoneJudge.evaluate(P047CapstoneSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testEndToEndReportAndCacheGrowth() throws {
    let request = try P047CapstoneFixture.makeRequest(
      maxNewTokens: 4, seed: 47, includeMetalVerification: false)
    let report = try P047CapstoneSolution.run(request)
    try P047CapstoneContract.validate(report, for: request)
    XCTAssertEqual(report.promptTokenIDs, [1, 2, 3, 5, 4, 6])
    XCTAssertTrue(report.generatedTokenIDs.allSatisfy { (0..<7).contains($0) })
    XCTAssertEqual(
      report.finalCacheCounts,
      Array(repeating: report.promptTokenIDs.count + max(0, report.generatedTokenIDs.count - 1), count: 2))
    XCTAssertGreaterThan(report.modelWeightBytes, 0)
    XCTAssertGreaterThan(report.allocatedKVCacheBytes, 0)
    XCTAssertEqual(report.generationBackend, "CPU reference backend")
  }

  func testTokenizerRoundTripAndVocabularyCompatibility() throws {
    let tokenizer = try P047CapstoneFixture.makeTokenizer()
    let model = try EducationalMiniModelFixture.make()
    XCTAssertEqual(tokenizer.tokensByID.keys.sorted(), Array(0..<model.vocabularySize))
    let ids = try P037ByteBPESolution.encode(
      tokenizer: tokenizer,
      text: P047CapstoneFixture.defaultPrompt,
      options: BPEEncodingOptions(addBeginningOfSequence: true))
    let bytes = try P037ByteBPESolution.decodeBytes(
      tokenizer: tokenizer,
      tokenIDs: ids,
      skipSpecialTokens: true)
    XCTAssertEqual(bytes, Array(P047CapstoneFixture.defaultPrompt.utf8))
  }

  func testJudgeRejectsDisconnectedPlaceholder() {
    let report = P047CapstoneJudge.evaluate { request in
      try P047CapstoneContract.validate(request)
      return CapstoneReport(
        prompt: request.prompt,
        promptTokenIDs: [],
        generatedTokenIDs: [],
        generatedBytes: [],
        rendering: .text(""),
        stopReason: .maximumTokenCount,
        timings: [],
        timeToFirstTokenNanoseconds: nil,
        decodeTokensPerSecond: nil,
        finalCacheCounts: [],
        modelWeightBytes: 0,
        allocatedKVCacheBytes: 0,
        prefillArenaBytes: 0,
        decodeArenaBytes: 0,
        generationBackend: "CPU reference backend",
        weightFormat: "Float32 row-major",
        keyValueFormat: "Float32 contiguous KV cache",
        metalVerification: CapstoneMetalVerification(
          label: P047CapstoneContract.metalVerificationLabel,
          status: .notRequested,
          captures: [],
          resources: nil),
        optimizationComparison: CapstoneOptimizationComparison(
          name: "none",
          baselineDispatchCount: 1,
          optimizedDispatchCount: 1,
          baselineLogicalBytes: 0,
          optimizedLogicalBytes: 0,
          basis: "none"),
        rejectedOptimization: CapstoneRejectedOptimization(name: "none", evidence: "none"),
        limitations: [])
    }
    XCTAssertFalse(report.isPassing)
  }
}
