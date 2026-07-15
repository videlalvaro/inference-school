import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P033FusedQ4GEMVMetalTests: XCTestCase {
  func testCanonicalMetalUnpacksInsideActualKernel() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let pipeline = try P033FusedQ4GEMVSolution.makeMetalPipeline()
    let report = P033FusedQ4GEMVJudge.evaluate(pipeline.multiply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testMetalResultReportsNoMaterializedFloatWeights() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let weights = try FloatTensor(
      (0..<33).map { Float(($0 % 11) - 5) / 3 }, shape: [3, 11])
    let quantized = try P033FusedQ4GEMVSolution.quantize(weights, groupSize: 4)
    let input = try FloatTensor((0..<11).map { Float($0 - 5) / 6 }, shape: [11])
    let result = try P033FusedQ4GEMVSolution.makeMetalPipeline().multiply(quantized, input)
    XCTAssertEqual(result.logicalWeightBytes, quantized.allocatedBytes)
    XCTAssertEqual(result.temporaryWeightBytes, 0)
    XCTAssertEqual(result.output.shape, [3])
  }
}