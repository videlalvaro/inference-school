import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P033FusedQ4GEMVTests: XCTestCase {
  func testCanonicalCPUPassesJudge() {
    let report = P033FusedQ4GEMVJudge.evaluate(P033FusedQ4GEMVSolution.multiply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsFullFloatWeightTemporary() {
    let report = P033FusedQ4GEMVJudge.evaluate { weights, input in
      let staged = try P032DequantizeThenGEMVSolution.multiply(weights, input)
      return FusedQ4GEMVResult(
        output: staged.output,
        logicalWeightBytes: weights.allocatedBytes,
        temporaryWeightBytes: staged.temporaryWeightBytes)
    }
    XCTAssertFalse(report.isPassing)
  }

  func testQuantizerUsesCanonicalFormatAndIncludesMetadataBytes() throws {
    let weights = try FloatTensor([-1, 0, 1, 2, -2], shape: [1, 5])
    let quantized = try P033FusedQ4GEMVSolution.quantize(weights, groupSize: 3)
    XCTAssertEqual(quantized.format, .signedTwosComplementLowNibbleFirst)
    XCTAssertEqual(quantized.packedValueBytes, 3)
    XCTAssertEqual(quantized.scaleBytes, 8)
    XCTAssertEqual(quantized.allocatedBytes, 11)
  }
}