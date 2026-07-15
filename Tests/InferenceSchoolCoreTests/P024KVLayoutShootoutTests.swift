import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P024KVLayoutShootoutTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P024KVLayoutShootoutJudge.evaluate(P024KVLayoutShootoutSolution.run)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.passedCaseCount, 3)
  }

  func testJudgeRejectsTreatingBothLayoutsAsTokenMajor() {
    let report = P024KVLayoutShootoutJudge.evaluate { logical, configuration, layer, head in
      let descriptor = KVLayoutDescriptor(kind: .tokenMajor, configuration: configuration)
      var offsets: [Int] = []
      for slot in 0..<configuration.capacity {
        for feature in 0..<configuration.headDimension {
          offsets.append(try descriptor.offset(
            layer: layer, slot: slot, head: head, feature: feature))
        }
      }
      let trace = KVAccessTrace(
        offsets: offsets,
        contiguousReadSpans: configuration.capacity,
        bytesRead: offsets.count * MemoryLayout<Float>.stride)
      return KVLayoutShootoutResult(
        tokenMajorRoundTrip: logical,
        headMajorRoundTrip: logical,
        tokenMajorTrace: trace,
        headMajorTrace: trace)
    }
    XCTAssertFalse(report.isPassing)
  }
}
