import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P040AutoregressiveDecodeTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P040AutoregressiveDecodeJudge.evaluate(P040AutoregressiveDecodeSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsClaimThatPriorTokensWereReprojected() {
    let report = P040AutoregressiveDecodeJudge.evaluate { request, cache, generator in
      let result = try P040AutoregressiveDecodeSolution.run(
        request, cache: cache, generator: &generator)
      return AutoregressiveDecodeResult(
        inputTokenID: result.inputTokenID,
        logicalPosition: result.logicalPosition,
        selectedNextTokenID: result.selectedNextTokenID,
        sampling: result.sampling,
        finalResidual: result.finalResidual,
        finalHidden: result.finalHidden,
        logits: result.logits,
        layers: result.layers,
        cacheCountsBefore: result.cacheCountsBefore,
        cacheCountsAfter: result.cacheCountsAfter,
        cachePositions: result.cachePositions,
        work: MiniDecoderWorkModel(
          projectionFLOPs: result.work.projectionFLOPs,
          attentionFLOPs: result.work.attentionFLOPs,
          estimatedWeightBytesRead: result.work.estimatedWeightBytesRead,
          cacheBytesWritten: result.work.cacheBytesWritten,
          keyValueProjectionInputTokens: result.work.keyValueProjectionInputTokens,
          priorKeyValueTokensReprojected: 1))
    }
    XCTAssertFalse(report.isPassing)
  }

  func testGenerationSessionIsDeterministicAndGrowsCacheOncePerDecodeStep() throws {
    let model = try EducationalMiniModelFixture.make()
    let strategy = SamplingStrategy.stochastic(
      SamplingConfiguration(temperature: 0.75, topK: 4, topP: 0.9))
    let first = try MiniDecoderGenerationSession(
      model: model,
      cacheCapacity: 12,
      samplingStrategy: strategy,
      seed: 99)
    let second = try MiniDecoderGenerationSession(
      model: model,
      cacheCapacity: 12,
      samplingStrategy: strategy,
      seed: 99)
    let firstResult = try first.generate(
      promptTokenIDs: EducationalMiniModelFixture.defaultPrompt,
      maxNewTokens: 4)
    let secondResult = try second.generate(
      promptTokenIDs: EducationalMiniModelFixture.defaultPrompt,
      maxNewTokens: 4)
    XCTAssertEqual(firstResult.generatedTokenIDs, secondResult.generatedTokenIDs)
    XCTAssertEqual(first.generator, second.generator)
    XCTAssertEqual(firstResult.decodeSteps.count, 3)
    for layer in 0..<model.layerCount {
      XCTAssertEqual(try first.cache.count(layer: layer), 6)
    }
  }

  func testGenerationStopsAtConfiguredEOS() throws {
    let model = try EducationalMiniModelFixture.make()
    let probeCache = ContiguousKVCache(
      configuration: try model.cacheConfiguration(capacity: 8))
    let prefill = try P039PromptPrefillSolution.run(
      PromptPrefillRequest(
        model: model, tokenIDs: EducationalMiniModelFixture.defaultPrompt),
      cache: probeCache)
    var generator = SeededGenerator(seed: 7)
    let firstToken = try P038LogitsSamplingSolution.sample(
      logits: prefill.logits.storage,
      strategy: .greedy,
      generator: &generator).selectedToken
    let session = try MiniDecoderGenerationSession(
      model: model,
      cacheCapacity: 8,
      samplingStrategy: .greedy,
      seed: 7,
      endOfSequenceTokenID: firstToken)
    let result = try session.generate(
      promptTokenIDs: EducationalMiniModelFixture.defaultPrompt,
      maxNewTokens: 5)
    XCTAssertEqual(result.generatedTokenIDs, [firstToken])
    XCTAssertEqual(result.stopReason, .endOfSequence)
    XCTAssertTrue(result.decodeSteps.isEmpty)
  }
}