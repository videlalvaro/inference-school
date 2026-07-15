import Foundation

public struct KVHeadByteComparison: Sendable, Equatable {
  public let mhaBytes: Int
  public let mqaBytes: Int
  public let gqaBytes: Int

  public init(mhaBytes: Int, mqaBytes: Int, gqaBytes: Int) {
    self.mhaBytes = mhaBytes
    self.mqaBytes = mqaBytes
    self.gqaBytes = gqaBytes
  }
}

public enum KVHeadMemoryModel {
  public static func bytes(
    layerCount: Int,
    tokenCount: Int,
    keyValueHeadCount: Int,
    headDimension: Int,
    scalarBytes: Int = MemoryLayout<Float>.stride
  ) throws -> Int {
    let configuration = try KVCacheConfiguration(
      layerCount: layerCount,
      keyValueHeadCount: keyValueHeadCount,
      headDimension: headDimension,
      capacity: tokenCount)
    let (bytes, overflow) = configuration.elementsPerTensor.multipliedReportingOverflow(
      by: 2 * scalarBytes)
    guard !overflow else { throw KVCacheError.storageSizeOverflow }
    return bytes
  }

  public static func compare(
    layerCount: Int,
    tokenCount: Int,
    queryHeadCount: Int,
    gqaHeadCount: Int,
    headDimension: Int,
    scalarBytes: Int = MemoryLayout<Float>.stride
  ) throws -> KVHeadByteComparison {
    _ = try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: gqaHeadCount,
      headDimension: headDimension)
    return KVHeadByteComparison(
      mhaBytes: try bytes(
        layerCount: layerCount, tokenCount: tokenCount,
        keyValueHeadCount: queryHeadCount, headDimension: headDimension,
        scalarBytes: scalarBytes),
      mqaBytes: try bytes(
        layerCount: layerCount, tokenCount: tokenCount,
        keyValueHeadCount: 1, headDimension: headDimension, scalarBytes: scalarBytes),
      gqaBytes: try bytes(
        layerCount: layerCount, tokenCount: tokenCount,
        keyValueHeadCount: gqaHeadCount, headDimension: headDimension,
        scalarBytes: scalarBytes))
  }
}

public struct SharedKVHeadsResult: Sendable, Equatable {
  public let attention: CachedAttentionResult
  public let bytes: KVHeadByteComparison

  public init(attention: CachedAttentionResult, bytes: KVHeadByteComparison) {
    self.attention = attention
    self.bytes = bytes
  }
}

public typealias SharedKVHeadsImplementation = (
  _ request: CachedAttentionRequest
) throws -> SharedKVHeadsResult

public enum P025SharedKVHeadsJudge {
  public static func evaluate(_ implementation: SharedKVHeadsImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let cache = try KVCacheConfiguration(
        layerCount: 2, keyValueHeadCount: 2, headDimension: 2, capacity: 3)
      let request = try CachedAttentionRequest(
        cacheConfiguration: cache,
        queryHeadCount: 4,
        layer: 0,
        firstLogicalPosition: 5,
        queryLogicalPosition: 7,
        query: FloatTensor([
          1, 0,
          0, 1,
          1, 1,
          -1, 1,
        ], shape: [4, 2]),
        keys: FloatTensor([
          1, 0, 0, 1,
          0, 1, 1, 0,
          1, 1, -1, 1,
        ], shape: [3, 2, 2]),
        values: FloatTensor([
          1, 2, 10, 20,
          3, 4, 30, 40,
          5, 6, 50, 60,
        ], shape: [3, 2, 2]))
      let actual = try implementation(request)
      let expected = try P023CachedAttentionJudge.oracle(request)
      if AttentionJudgeOracle.approximatelyEqual(actual.attention.output, expected) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "contiguous query groups use division mapping",
          message: "expected q heads [0,1] to share KV head 0 and [2,3] to share KV head 1"))
      }
      if actual.bytes == KVHeadByteComparison(mhaBytes: 384, mqaBytes: 96, gqaBytes: 192) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "MHA MQA GQA byte model",
          message: "expected 2*L*T*Hkv*D*4 bytes for each architecture"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "shared KV heads", message: error.localizedDescription))
    }
    do {
      _ = try AttentionConfiguration(queryHeadCount: 3, keyValueHeadCount: 2, headDimension: 4)
      failures.append(JudgeFailure(
        caseName: "reject invalid divisibility", message: "3 query heads cannot share 2 KV heads"))
    } catch {
      passed += 1
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 3, failures: failures)
  }
}
