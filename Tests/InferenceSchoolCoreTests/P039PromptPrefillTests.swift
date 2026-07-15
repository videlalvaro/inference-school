import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P039PromptPrefillTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P039PromptPrefillJudge.evaluate(P039PromptPrefillSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsMissingCacheWrites() {
    let report = P039PromptPrefillJudge.evaluate { request, _ in
      let privateCache = ContiguousKVCache(
        configuration: try request.model.cacheConfiguration(capacity: request.tokenIDs.count))
      return try P039PromptPrefillSolution.run(request, cache: privateCache)
    }
    XCTAssertFalse(report.isPassing)
  }

  func testModelRejectsWrongOutputShapeAndNonFiniteEmbedding() throws {
    let valid = try EducationalMiniModelFixture.make(layerCount: 1)
    XCTAssertThrowsError(try MiniDecoderModel(
      vocabularySize: valid.vocabularySize,
      configuration: valid.configuration,
      tokenEmbedding: FloatTensor(
        Array(repeating: 0, count: valid.tokenEmbedding.elementCount),
        shape: valid.tokenEmbedding.shape),
      blocks: valid.blocks,
      finalNormGamma: valid.finalNormGamma,
      outputProjection: .independent(FloatTensor([0], shape: [1, 1]))))
    var values = valid.tokenEmbedding.storage
    values[3] = .nan
    XCTAssertThrowsError(try MiniDecoderModel(
      vocabularySize: valid.vocabularySize,
      configuration: valid.configuration,
      tokenEmbedding: FloatTensor(values, shape: valid.tokenEmbedding.shape),
      blocks: valid.blocks,
      finalNormGamma: valid.finalNormGamma,
      outputProjection: .tiedEmbedding))
  }

  func testFingerprintChangesWithLayerWeights() throws {
    let oneLayer = try EducationalMiniModelFixture.make(layerCount: 1)
    let twoLayers = try EducationalMiniModelFixture.make(layerCount: 2)
    XCTAssertNotEqual(oneLayer.fingerprint, twoLayers.fingerprint)
    XCTAssertEqual(oneLayer.fingerprint, try EducationalMiniModelFixture.make(layerCount: 1).fingerprint)
  }
}