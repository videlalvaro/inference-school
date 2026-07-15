public typealias Q4PackingImplementation = (
  _ values: [Int8],
  _ outputChannels: Int,
  _ inputChannels: Int,
  _ groupSize: Int,
  _ scales: [Float]
) throws -> GroupwiseQ4WeightMatrix

public enum P031PackQ4Judge {
  public static func evaluate(_ implementation: Q4PackingImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let values: [Int8] = [-8, -7, -1, 0, 1, 7, 3]
      let packed = try implementation(values, 1, 7, 3, [0.5, 0.25, 0.125])
      if packed.packedValues == [0x98, 0x0f, 0x71, 0x03],
        unpack(packed) == values,
        packed.groupsPerOutputChannel == 3,
        packed.allocatedBytes == 16
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "byte-exact signed Q4 fixture",
          message: "expected low nibble first two's-complement bytes [98,0f,71,03] and zero padding"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "byte-exact signed Q4 fixture", message: error.localizedDescription))
    }

    do {
      let values: [Int8] = [-8, 7, 1, -1, 2, -2]
      let packed = try implementation(values, 2, 3, 2, [1, 2, 3, 4])
      if packed.packedValues == [0x78, 0xf1, 0xe2], unpack(packed) == values,
        packed.scaleIndex(outputChannel: 1, inputChannel: 2) == 3
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "row-major stream and scale alignment",
          message: "odd row width must not change logical row-major indexing or [out,group] scales"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "row-major stream", message: error.localizedDescription))
    }

    do {
      _ = try implementation([8], 1, 1, 1, [1])
      failures.append(JudgeFailure(
        caseName: "reject out-of-range Q4", message: "expected value 8 to throw"))
    } catch { passed += 1 }
    do {
      _ = try implementation([1, 2], 1, 2, 1, [1])
      failures.append(JudgeFailure(
        caseName: "reject scale count", message: "expected missing group scale to throw"))
    } catch { passed += 1 }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 4, failures: failures)
  }

  private static func unpack(_ matrix: GroupwiseQ4WeightMatrix) -> [Int8] {
    (0..<matrix.outputChannels).flatMap { row in
      (0..<matrix.inputChannels).map { column in
        matrix.quantizedValue(outputChannel: row, inputChannel: column)
      }
    }
  }
}