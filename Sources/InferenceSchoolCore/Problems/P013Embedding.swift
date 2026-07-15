import Foundation

public enum EmbeddingLookupError: Error, Equatable, LocalizedError {
  case emptyVocabulary
  case emptyEmbeddingDimension
  case tokenOutOfRange(index: Int, token: Int, vocabularySize: Int)

  public var errorDescription: String? {
    switch self {
    case .emptyVocabulary:
      "The embedding vocabulary must contain at least one row."
    case .emptyEmbeddingDimension:
      "The embedding dimension must be positive."
    case .tokenOutOfRange(let index, let token, let vocabularySize):
      "Token at index \(index) is \(token), outside 0..<\(vocabularySize)."
    }
  }
}

public struct EmbeddingLookupResult: Sendable, Equatable {
  public let embeddings: FloatTensor
  public let logits: FloatTensor

  public init(embeddings: FloatTensor, logits: FloatTensor) {
    self.embeddings = embeddings
    self.logits = logits
  }
}

public typealias EmbeddingLookupImplementation = (
  _ tokenIDs: [Int],
  _ table: FloatTensor
) throws -> EmbeddingLookupResult

public enum EmbeddingLookupContract {
  public static func validate(tokenIDs: [Int], table: FloatTensor) throws {
    guard table.rank == 2 else {
      throw TensorError.rankMismatch(expected: 2, actual: table.rank)
    }
    guard table.shape[0] > 0 else { throw EmbeddingLookupError.emptyVocabulary }
    guard table.shape[1] > 0 else { throw EmbeddingLookupError.emptyEmbeddingDimension }
    for (index, token) in tokenIDs.enumerated() where token < 0 || token >= table.shape[0] {
      throw EmbeddingLookupError.tokenOutOfRange(
        index: index,
        token: token,
        vocabularySize: table.shape[0]
      )
    }
  }
}

public enum P013EmbeddingJudge {
  public static let absoluteTolerance: Float = 2e-5
  public static let relativeTolerance: Float = 4e-5

  private struct ValueCase {
    let name: String
    let tokenIDs: [Int]
    let table: FloatTensor
  }

  public static func evaluate(_ implementation: EmbeddingLookupImplementation) -> JudgeReport {
    let cases: [ValueCase]
    do {
      cases = [
        ValueCase(
          name: "gather order and tied logits",
          tokenIDs: [2, 0],
          table: try FloatTensor([1, 2, 3, 4, 5, 6], shape: [3, 2])
        ),
        ValueCase(
          name: "duplicate tokens and mixed signs",
          tokenIDs: [1, 1, 3],
          table: try FloatTensor(
            [
              0.5, -1, 2,
              -2, 0.25, 1,
              1.5, 2, -0.5,
              0, -3, 0.75,
            ], shape: [4, 3])
        ),
        ValueCase(
          name: "empty sequence",
          tokenIDs: [],
          table: try FloatTensor([1, -1, 2, -2], shape: [2, 2])
        ),
      ]
    } catch {
      return JudgeReport(
        passedCaseCount: 0,
        totalCaseCount: 7,
        failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)]
      )
    }

    var failures: [JudgeFailure] = []
    var passed = 0
    for testCase in cases {
      do {
        let actual = try implementation(testCase.tokenIDs, testCase.table)
        let expected = try reference(tokenIDs: testCase.tokenIDs, table: testCase.table)
        if actual.embeddings.shape == expected.embeddings.shape,
          actual.logits.shape == expected.logits.shape,
          approximatelyEqual(actual.embeddings.storage, expected.embeddings.storage),
          approximatelyEqual(actual.logits.storage, expected.logits.storage)
        {
          passed += 1
        } else {
          failures.append(
            JudgeFailure(
              caseName: testCase.name,
              message:
                "expected embeddings \(expected.embeddings.storage) and logits \(expected.logits.storage), received shapes \(actual.embeddings.shape)/\(actual.logits.shape) and values \(actual.embeddings.storage)/\(actual.logits.storage)"
            ))
        }
      } catch {
        failures.append(
          JudgeFailure(
            caseName: testCase.name,
            message: "unexpected error: \(error.localizedDescription)"
          ))
      }
    }

    passed += expectError(name: "reject rank-one table", failures: &failures) {
      _ = try implementation([0], FloatTensor([1, 2], shape: [2]))
    }
    passed += expectError(name: "reject negative token", failures: &failures) {
      _ = try implementation([-1], FloatTensor([1, 2], shape: [1, 2]))
    }
    passed += expectError(name: "reject token at vocabulary bound", failures: &failures) {
      _ = try implementation([2], FloatTensor([1, 2, 3, 4], shape: [2, 2]))
    }
    passed += expectError(name: "reject empty embedding width", failures: &failures) {
      _ = try implementation([], FloatTensor([], shape: [2, 0]))
    }

    return JudgeReport(
      passedCaseCount: passed,
      totalCaseCount: cases.count + 4,
      failures: failures
    )
  }

  private static func reference(tokenIDs: [Int], table: FloatTensor) throws -> EmbeddingLookupResult
  {
    let vocabularySize = table.shape[0]
    let embeddingDimension = table.shape[1]
    var gathered: [Float] = []
    gathered.reserveCapacity(tokenIDs.count * embeddingDimension)
    for token in tokenIDs {
      let rowStart = token * embeddingDimension
      gathered.append(contentsOf: table.storage[rowStart..<(rowStart + embeddingDimension)])
    }

    var logits = Array(repeating: Float.zero, count: tokenIDs.count * vocabularySize)
    for sequence in tokenIDs.indices {
      for vocabulary in 0..<vocabularySize {
        var sum = 0.0
        for feature in 0..<embeddingDimension {
          sum +=
            Double(gathered[sequence * embeddingDimension + feature])
            * Double(table.storage[vocabulary * embeddingDimension + feature])
        }
        logits[sequence * vocabularySize + vocabulary] = Float(sum)
      }
    }
    return EmbeddingLookupResult(
      embeddings: try FloatTensor(gathered, shape: [tokenIDs.count, embeddingDimension]),
      logits: try FloatTensor(logits, shape: [tokenIDs.count, vocabularySize])
    )
  }

  private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
    lhs.count == rhs.count
      && zip(lhs, rhs).allSatisfy { actual, expected in
        abs(actual - expected) <= absoluteTolerance + relativeTolerance * abs(expected)
      }
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(
        JudgeFailure(
          caseName: name,
          message: "expected an error, but the implementation returned"
        ))
      return 0
    } catch {
      return 1
    }
  }
}
