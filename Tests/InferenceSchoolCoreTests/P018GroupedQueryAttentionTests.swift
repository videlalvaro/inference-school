import Foundation
import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P018GroupedQueryAttentionTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let r = P018GroupedQueryAttentionJudge.evaluate(P018GroupedQueryAttentionSolution.apply)
    XCTAssertTrue(r.isPassing, r.failures.map(\.message).joined(separator: "\n"))
  }
  func testJudgeRejectsModuloBasedGQAHeadMapping() {
    let report = P018GroupedQueryAttentionJudge.evaluate(moduloMappedAttention)

    XCTAssertFalse(report.isPassing)
    XCTAssertTrue(report.failures.contains { $0.caseName == "GQA two query heads per KV head" })
  }
  func testConfigurationRejectsNonDivisibleHeads() {
    XCTAssertThrowsError(
      try AttentionConfiguration(queryHeadCount: 3, keyValueHeadCount: 2, headDimension: 4))
  }
  func testCacheBytesShrinkWithKVHeads() throws {
    let mha = try KVCacheByteModel(sequenceLength: 128, keyValueHeadCount: 8, headDimension: 64)
    let gqa = try KVCacheByteModel(sequenceLength: 128, keyValueHeadCount: 2, headDimension: 64)
    XCTAssertEqual(mha.bytes, 4 * gqa.bytes)
  }
}

private func moduloMappedAttention(
  _ queries: FloatTensor,
  _ keys: FloatTensor,
  _ values: FloatTensor,
  _ configuration: AttentionConfiguration
) throws -> FloatTensor {
  let input = try P018GroupedQueryAttentionContract.validate(
    queries, keys, values, configuration)
  var output = Array(repeating: Float.zero, count: queries.elementCount)
  let scale = 1 / sqrt(Float(configuration.headDimension))

  for query in 0..<input.queryLength {
    let queryPosition = configuration.queryPositionOffset + query
    for queryHead in 0..<configuration.queryHeadCount {
      let keyValueHead = queryHead % configuration.keyValueHeadCount
      var visibleKeys: [Int] = []
      var scores: [Float] = []

      for key in 0..<input.keyValueLength
      where configuration.keyPositionOffset + key <= queryPosition {
        var dot: Float = 0
        for feature in 0..<configuration.headDimension {
          dot +=
            queries.storage[
              input.queryOffset(sequence: query, head: queryHead, feature: feature)
            ]
            * keys.storage[
              input.keyValueOffset(sequence: key, head: keyValueHead, feature: feature)
            ]
        }
        visibleKeys.append(key)
        scores.append(dot * scale)
      }

      guard let maximum = scores.max() else {
        throw AttentionError.noVisibleKeys(queryPosition: queryPosition)
      }
      let exponentials = scores.map { exp($0 - maximum) }
      let denominator = exponentials.reduce(0, +)
      for feature in 0..<configuration.headDimension {
        for (index, key) in visibleKeys.enumerated() {
          output[input.queryOffset(sequence: query, head: queryHead, feature: feature)] +=
            exponentials[index] / denominator
            * values.storage[
              input.keyValueOffset(sequence: key, head: keyValueHead, feature: feature)
            ]
        }
      }
    }
  }

  return try FloatTensor(output, shape: queries.shape)
}
