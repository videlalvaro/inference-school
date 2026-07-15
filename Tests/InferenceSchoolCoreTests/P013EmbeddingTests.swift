import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P013EmbeddingTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P013EmbeddingJudge.evaluate(P013EmbeddingSolution.apply)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsLookupWithoutTiedUnembedding() {
    let report = P013EmbeddingJudge.evaluate { tokenIDs, table in
      try EmbeddingLookupContract.validate(tokenIDs: tokenIDs, table: table)
      var values: [Float] = []
      for token in tokenIDs {
        let start = token * table.shape[1]
        values.append(contentsOf: table.storage[start..<(start + table.shape[1])])
      }
      return EmbeddingLookupResult(
        embeddings: try FloatTensor(values, shape: [tokenIDs.count, table.shape[1]]),
        logits: try FloatTensor(
          Array(repeating: 0, count: tokenIDs.count * table.shape[0]),
          shape: [tokenIDs.count, table.shape[0]]
        )
      )
    }
    XCTAssertFalse(report.isPassing)
  }

  func testTiedTableDefinesBothDirections() throws {
    let table = try FloatTensor([1, 0, 0, 2], shape: [2, 2])
    let result = try P013EmbeddingSolution.apply(tokenIDs: [1], table: table)
    XCTAssertEqual(result.embeddings.storage, [0, 2])
    XCTAssertEqual(result.logits.storage, [0, 4])
  }
}
