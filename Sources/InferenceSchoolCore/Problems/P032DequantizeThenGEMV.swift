public struct DequantizeThenGEMVResult: Sendable, Equatable {
  public let output: FloatTensor
  public let materializedWeights: FloatTensor
  public let temporaryWeightBytes: Int

  public init(
    output: FloatTensor,
    materializedWeights: FloatTensor,
    temporaryWeightBytes: Int
  ) {
    self.output = output
    self.materializedWeights = materializedWeights
    self.temporaryWeightBytes = temporaryWeightBytes
  }
}

public typealias DequantizeThenGEMVImplementation = (
  _ weights: GroupwiseQ4WeightMatrix,
  _ input: FloatTensor
) throws -> DequantizeThenGEMVResult

public enum P032DequantizeThenGEMVJudge {
  public static func evaluate(_ implementation: DequantizeThenGEMVImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let fixture = try makeFixture()
      let actual = try implementation(fixture.weights, fixture.input)
      let expectedWeights = dequantize(fixture.weights)
      let expectedOutput = gemv(
        expectedWeights,
        rows: fixture.weights.outputChannels,
        columns: fixture.weights.inputChannels,
        input: fixture.input.storage)
      if actual.materializedWeights.shape == fixture.weights.shape,
        approximatelyEqual(actual.materializedWeights.storage, expectedWeights),
        actual.output.shape == [fixture.weights.outputChannels],
        approximatelyEqual(actual.output.storage, expectedOutput)
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "materialize Q4 then GEMV",
          message: "materialized weights or GEMV output differed from the independent oracle"))
      }
      if actual.temporaryWeightBytes
        == fixture.weights.logicalValueCount * MemoryLayout<Float>.stride
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "temporary Float32 weight bytes",
          message: "baseline must report one full [out,in] Float32 temporary"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "staged Q4 GEMV", message: error.localizedDescription))
    }

    do {
      let fixture = try makeFixture()
      _ = try implementation(fixture.weights, FloatTensor([1, 2], shape: [1, 2]))
      failures.append(JudgeFailure(caseName: "reject input rank", message: "expected rank-1 input"))
    } catch { passed += 1 }
    do {
      let fixture = try makeFixture()
      _ = try implementation(fixture.weights, FloatTensor([1, 2], shape: [2]))
      failures.append(JudgeFailure(
        caseName: "reject input width", message: "expected input width mismatch"))
    } catch { passed += 1 }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }

  private static func makeFixture() throws -> (
    weights: GroupwiseQ4WeightMatrix, input: FloatTensor
  ) {
    (
      try GroupwiseQ4WeightMatrix(
        outputChannels: 2,
        inputChannels: 5,
        groupSize: 3,
        packedValues: [0xc8, 0x30, 0x17, 0x2f, 0x0e],
        scales: [0.25, 0.5, 0.1, 0.2]),
      try FloatTensor([1, -2, 0.5, 1.5, -1], shape: [5])
    )
  }

  private static func dequantize(_ weights: GroupwiseQ4WeightMatrix) -> [Float] {
    (0..<weights.outputChannels).flatMap { output in
      (0..<weights.inputChannels).map { input in
        Float(weights.quantizedValue(outputChannel: output, inputChannel: input))
          * weights.scales[weights.scaleIndex(outputChannel: output, inputChannel: input)]
      }
    }
  }

  private static func gemv(
    _ weights: [Float], rows: Int, columns: Int, input: [Float]
  ) -> [Float] {
    (0..<rows).map { row in
      Float((0..<columns).reduce(0.0) { partial, column in
        partial + Double(weights[row * columns + column]) * Double(input[column])
      })
    }
  }

  private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
    lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { left, right in
      abs(left - right) <= 2e-5 * max(1, abs(left), abs(right))
    }
  }
}