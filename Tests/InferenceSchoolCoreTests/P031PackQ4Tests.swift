import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P031PackQ4Tests: XCTestCase {
  func testCanonicalSolutionPassesByteExactJudge() {
    let report = P031PackQ4Judge.evaluate(P031PackQ4Solution.pack)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsHighNibbleFirstPacking() {
    let report = P031PackQ4Judge.evaluate { values, rows, columns, groupSize, scales in
      var bytes = Array(repeating: UInt8.zero, count: values.count / 2 + values.count % 2)
      for (index, value) in values.enumerated() {
        let nibble = UInt8(bitPattern: value) & 0x0f
        bytes[index / 2] |= index.isMultiple(of: 2) ? nibble << 4 : nibble
      }
      if values.count % 2 == 1 { bytes[bytes.count - 1] &= 0x0f }
      return try GroupwiseQ4WeightMatrix(
        outputChannels: rows, inputChannels: columns, groupSize: groupSize,
        packedValues: bytes, scales: scales)
    }
    XCTAssertFalse(report.isPassing)
  }

  func testOddCountRejectsNonzeroPaddingNibble() {
    XCTAssertThrowsError(try GroupwiseQ4WeightMatrix(
      outputChannels: 1, inputChannels: 1, groupSize: 1,
      packedValues: [0xf1], scales: [1]))
  }
}