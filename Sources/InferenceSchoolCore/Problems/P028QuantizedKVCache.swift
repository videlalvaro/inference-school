import Foundation

public struct QuantizedKVCacheResult: Sendable, Equatable {
  public let attentionOutput: FloatTensor
  public let dequantizedKeys: FloatTensor
  public let dequantizedValues: FloatTensor
  public let keyScales: [Float]
  public let valueScales: [Float]
  public let allocatedBytes: Int
  public let maximumKeyError: Float
  public let maximumValueError: Float

  public init(
    attentionOutput: FloatTensor,
    dequantizedKeys: FloatTensor,
    dequantizedValues: FloatTensor,
    keyScales: [Float],
    valueScales: [Float],
    allocatedBytes: Int,
    maximumKeyError: Float,
    maximumValueError: Float
  ) {
    self.attentionOutput = attentionOutput
    self.dequantizedKeys = dequantizedKeys
    self.dequantizedValues = dequantizedValues
    self.keyScales = keyScales
    self.valueScales = valueScales
    self.allocatedBytes = allocatedBytes
    self.maximumKeyError = maximumKeyError
    self.maximumValueError = maximumValueError
  }
}

public typealias QuantizedKVCacheImplementation = (
  _ request: CachedAttentionRequest
) throws -> QuantizedKVCacheResult

public enum P028QuantizedKVCacheJudge {
  public static func evaluate(_ implementation: QuantizedKVCacheImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let configuration = try KVCacheConfiguration(
        layerCount: 1, keyValueHeadCount: 2, headDimension: 4, capacity: 3)
      let request = try CachedAttentionRequest(
        cacheConfiguration: configuration,
        queryHeadCount: 2,
        layer: 0,
        firstLogicalPosition: 8,
        queryLogicalPosition: 10,
        query: FloatTensor([
          0.5, -0.25, 1, 0.75,
          -0.75, 0.5, 0.25, 1,
        ], shape: [2, 4]),
        keys: FloatTensor([
          0.1, -0.2, 0.3, 0.4, 1.2, -1.0, 0.5, 0.25,
          -0.4, 0.6, 0.2, -0.1, 0.75, 0.5, -0.25, 1.0,
          0.9, -0.8, 0.7, -0.6, -1.1, 0.2, 0.4, 0.8,
        ], shape: [3, 2, 4]),
        values: FloatTensor([
          1, 2, 3, 4, 10, 20, 30, 40,
          2, 4, 6, 8, 12, 24, 36, 48,
          3, 6, 9, 12, 14, 28, 42, 56,
        ], shape: [3, 2, 4]))
      let actual = try implementation(request)
      let floatOutput = try P023CachedAttentionJudge.oracle(request)
      if AttentionJudgeOracle.approximatelyEqual(
        actual.attentionOutput,
        floatOutput,
        absoluteTolerance: 0.08,
        relativeTolerance: 0.015)
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "quantized cached attention stays near Float cache",
          message: "output exceeded the explicit INT8 error tolerance"))
      }
      if actual.dequantizedKeys.shape == request.keys.shape,
        actual.dequantizedValues.shape == request.values.shape,
        actual.maximumKeyError <= 0.005,
        actual.maximumValueError <= 0.23
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "per-vector quantize dequantize error",
          message: "round-trip shape or max error violates half-scale bounds"))
      }
      let expectedBytes = 2 * configuration.elementsPerTensor * MemoryLayout<Int8>.stride
        + 2 * configuration.layerCount * configuration.capacity
          * configuration.keyValueHeadCount * MemoryLayout<Float>.stride
      if actual.allocatedBytes == expectedBytes,
        actual.keyScales.count == 6,
        actual.valueScales.count == 6,
        actual.keyScales != actual.valueScales
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "INT8 bytes include independent K and V scale metadata",
          message: "expected quantized elements plus one Float scale per token/head/vector"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "quantized cache", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 3, failures: failures)
  }
}
