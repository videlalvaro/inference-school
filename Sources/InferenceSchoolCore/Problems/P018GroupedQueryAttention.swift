import Foundation

public struct KVCacheByteModel: Sendable, Equatable {
  public let sequenceLength: Int
  public let keyValueHeadCount: Int
  public let headDimension: Int
  public let bytesPerElement: Int
  public var bytes: Int { 2 * sequenceLength * keyValueHeadCount * headDimension * bytesPerElement }
  public init(
    sequenceLength: Int, keyValueHeadCount: Int, headDimension: Int,
    bytesPerElement: Int = MemoryLayout<Float>.stride
  ) throws {
    guard sequenceLength >= 0 else {
      throw TensorError.invalidDimension(axis: 0, value: sequenceLength)
    }
    guard keyValueHeadCount > 0 else {
      throw AttentionError.invalidHeadCount(name: "Key/value head count", value: keyValueHeadCount)
    }
    guard headDimension > 0 else { throw AttentionError.invalidHeadDimension(headDimension) }
    guard bytesPerElement > 0 else {
      throw TensorError.invalidDimension(axis: 3, value: bytesPerElement)
    }
    self.sequenceLength = sequenceLength
    self.keyValueHeadCount = keyValueHeadCount
    self.headDimension = headDimension
    self.bytesPerElement = bytesPerElement
  }
}

public enum P018GroupedQueryAttentionContract {
  public static func validate(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, _ c: AttentionConfiguration
  ) throws -> AttentionInput {
    let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
    try validateVisibleKeys(input)
    return input
  }
}

public enum P018GroupedQueryAttentionJudge {
  public static func evaluate(_ implementation: AttentionImplementation) -> JudgeReport {
    do {
      return evaluateAttentionImplementation(
        implementation,
        configurations: [
          try AttentionConfiguration(queryHeadCount: 4, keyValueHeadCount: 2, headDimension: 2),
          try AttentionConfiguration(queryHeadCount: 4, keyValueHeadCount: 1, headDimension: 2),
          try AttentionConfiguration(queryHeadCount: 2, keyValueHeadCount: 2, headDimension: 2),
        ],
        caseNames: [
          "GQA two query heads per KV head", "MQA shared KV head", "MHA one-to-one mapping",
        ])
    } catch {
      return JudgeReport(
        passedCaseCount: 0,
        totalCaseCount: 4,
        failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)]
      )
    }
  }
}
