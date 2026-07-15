public struct FusedQ4GEMVResult: Sendable, Equatable {
  public let output: FloatTensor
  public let logicalWeightBytes: Int
  public let temporaryWeightBytes: Int

  public init(output: FloatTensor, logicalWeightBytes: Int, temporaryWeightBytes: Int) {
    self.output = output
    self.logicalWeightBytes = logicalWeightBytes
    self.temporaryWeightBytes = temporaryWeightBytes
  }
}

public typealias FusedQ4GEMVImplementation = (
  _ weights: GroupwiseQ4WeightMatrix,
  _ input: FloatTensor
) throws -> FusedQ4GEMVResult

public enum P033FusedQ4GEMVJudge {
  public static func evaluate(_ implementation: FusedQ4GEMVImplementation) -> JudgeReport {
    let cases: [(String, GroupwiseQ4WeightMatrix, FloatTensor)]
    do {
      cases = [
        (
          "odd input width and tail group",
          try GroupwiseQ4WeightMatrix(
            outputChannels: 2, inputChannels: 5, groupSize: 3,
            packedValues: [0xc8, 0x30, 0x17, 0x2f, 0x0e],
            scales: [0.25, 0.5, 0.1, 0.2]),
          try FloatTensor([1, -2, 0.5, 1.5, -1], shape: [5])
        ),
        (
          "input and group exceed one reduction stride",
          try patternedWeights(rows: 3, columns: 259, groupSize: 257),
          try FloatTensor(
            (0..<259).map { Float(($0 % 13) - 6) / 7 }, shape: [259])
        ),
        (
          "zero output rows",
          try GroupwiseQ4WeightMatrix(
            outputChannels: 0, inputChannels: 5, groupSize: 4,
            packedValues: [], scales: []),
          try FloatTensor([1, 2, 3, 4, 5], shape: [5])
        ),
      ]
    } catch {
      return JudgeReport(
        passedCaseCount: 0,
        totalCaseCount: 5,
        failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)])
    }

    var failures: [JudgeFailure] = []
    var passed = 0
    for (name, weights, input) in cases {
      do {
        let actual = try implementation(weights, input)
        let expected = oracle(weights: weights, input: input.storage)
        if actual.output.shape == [weights.outputChannels],
          approximatelyEqual(actual.output.storage, expected),
          actual.logicalWeightBytes == weights.allocatedBytes,
          actual.temporaryWeightBytes == 0
        {
          passed += 1
        } else {
          failures.append(JudgeFailure(
            caseName: name,
            message: "expected fused output, exact packed+scale bytes, and zero full-weight temporary"))
        }
      } catch {
        failures.append(JudgeFailure(caseName: name, message: error.localizedDescription))
      }
    }

    do {
      _ = try implementation(cases[0].1, FloatTensor([1, 2], shape: [1, 2]))
      failures.append(JudgeFailure(caseName: "reject input rank", message: "expected rank-1 input"))
    } catch { passed += 1 }
    do {
      _ = try implementation(cases[0].1, FloatTensor([1, 2], shape: [2]))
      failures.append(JudgeFailure(caseName: "reject input width", message: "expected width error"))
    } catch { passed += 1 }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 5, failures: failures)
  }

  private static func patternedWeights(
    rows: Int, columns: Int, groupSize: Int
  ) throws -> GroupwiseQ4WeightMatrix {
    let values: [Int8] = (0..<(rows * columns)).map { Int8(($0 % 16) - 8) }
    var packed = Array(repeating: UInt8.zero, count: values.count / 2 + values.count % 2)
    for (index, value) in values.enumerated() {
      let nibble = UInt8(bitPattern: value) & 0x0f
      packed[index / 2] |= index.isMultiple(of: 2) ? nibble : nibble << 4
    }
    let groups = (columns + groupSize - 1) / groupSize
    let scales = (0..<(rows * groups)).map { Float($0 + 1) / 32 }
    return try GroupwiseQ4WeightMatrix(
      outputChannels: rows, inputChannels: columns, groupSize: groupSize,
      packedValues: packed, scales: scales)
  }

  private static func oracle(weights: GroupwiseQ4WeightMatrix, input: [Float]) -> [Float] {
    (0..<weights.outputChannels).map { output in
      Float((0..<weights.inputChannels).reduce(0.0) { partial, inputChannel in
        let quantized = weights.quantizedValue(
          outputChannel: output, inputChannel: inputChannel)
        let scale = weights.scales[weights.scaleIndex(
          outputChannel: output, inputChannel: inputChannel)]
        return partial + Double(quantized) * Double(scale) * Double(input[inputChannel])
      })
    }
  }

  private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
    lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { left, right in
      abs(left - right) <= 8e-5 * max(1, abs(left), abs(right))
    }
  }
}