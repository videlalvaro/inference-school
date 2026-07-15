import Foundation

public enum RoPEError: Error, Equatable, LocalizedError {
  case rotaryDimensionMustBeEven(Int)
  case rotaryDimensionExceedsHeadDimension(rotary: Int, head: Int)
  case invalidBase(Float)
  case headDimensionMismatch(queries: Int, keys: Int)
  case nonFiniteValue(tensor: String, linearIndex: Int)

  public var errorDescription: String? {
    switch self {
    case .rotaryDimensionMustBeEven(let value):
      "Rotary dimension must be positive and even; received \(value)."
    case .rotaryDimensionExceedsHeadDimension(let rotary, let head):
      "Rotary dimension \(rotary) exceeds head dimension \(head)."
    case .invalidBase(let value):
      "RoPE base must be finite and greater than one; received \(value)."
    case .headDimensionMismatch(let queries, let keys):
      "Query and key head dimensions must match; received \(queries) and \(keys)."
    case .nonFiniteValue(let tensor, let linearIndex):
      "\(tensor) contains a non-finite value at linear index \(linearIndex)."
    }
  }
}

public struct RoPEResult: Sendable, Equatable {
  public let queries: FloatTensor
  public let keys: FloatTensor

  public init(queries: FloatTensor, keys: FloatTensor) {
    self.queries = queries
    self.keys = keys
  }
}

public typealias RoPEImplementation = (
  _ queries: FloatTensor,
  _ keys: FloatTensor,
  _ rotaryDimension: Int,
  _ base: Float,
  _ queryPositionOffset: Int,
  _ keyPositionOffset: Int
) throws -> RoPEResult

public enum RoPEContract {
  public static func validate(
    queries: FloatTensor,
    keys: FloatTensor,
    rotaryDimension: Int,
    base: Float,
    queryPositionOffset: Int,
    keyPositionOffset: Int
  ) throws {
    guard queries.rank == 3 else {
      throw TensorError.rankMismatch(expected: 3, actual: queries.rank)
    }
    guard keys.rank == 3 else { throw TensorError.rankMismatch(expected: 3, actual: keys.rank) }
    guard queries.shape[2] == keys.shape[2] else {
      throw RoPEError.headDimensionMismatch(queries: queries.shape[2], keys: keys.shape[2])
    }
    guard rotaryDimension > 0, rotaryDimension.isMultiple(of: 2) else {
      throw RoPEError.rotaryDimensionMustBeEven(rotaryDimension)
    }
    guard rotaryDimension <= queries.shape[2] else {
      throw RoPEError.rotaryDimensionExceedsHeadDimension(
        rotary: rotaryDimension,
        head: queries.shape[2]
      )
    }
    guard base.isFinite, base > 1 else { throw RoPEError.invalidBase(base) }
    guard queryPositionOffset >= 0 else {
      throw AttentionError.invalidPositionOffset(
        name: "Query position offset", value: queryPositionOffset)
    }
    guard keyPositionOffset >= 0 else {
      throw AttentionError.invalidPositionOffset(
        name: "Key position offset", value: keyPositionOffset)
    }
    for (name, tensor) in [("Queries", queries), ("Keys", keys)] {
      if let index = tensor.storage.firstIndex(where: { !$0.isFinite }) {
        throw RoPEError.nonFiniteValue(tensor: name, linearIndex: index)
      }
    }
  }
}

public enum P015RoPEJudge {
  public static let absoluteTolerance: Float = 3e-5
  public static let relativeTolerance: Float = 5e-5

  public static func evaluate(_ implementation: RoPEImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0
    do {
      let cases = [
        (
          "positions zero and one",
          try FloatTensor([1, 2, 3, 4, -1, 0.5, 2, -3], shape: [2, 1, 4]),
          try FloatTensor([2, -1, 0.5, 3, 4, 1, -2, 0], shape: [2, 1, 4]),
          4, Float(10_000), 0, 0
        ),
        (
          "partial rotary with different offsets",
          try FloatTensor([1, 2, 3, 4, 9, 10], shape: [1, 1, 6]),
          try FloatTensor([-1, 1, 2, -2, 7, 8], shape: [1, 1, 6]),
          4, Float(100), 3, 7
        ),
      ]
      for testCase in cases {
        let actual = try implementation(
          testCase.1, testCase.2, testCase.3, testCase.4, testCase.5, testCase.6)
        let expected = try reference(
          queries: testCase.1,
          keys: testCase.2,
          rotaryDimension: testCase.3,
          base: testCase.4,
          queryOffset: testCase.5,
          keyOffset: testCase.6
        )
        if equal(actual.queries, expected.queries), equal(actual.keys, expected.keys) {
          passed += 1
        } else {
          failures.append(
            JudgeFailure(
              caseName: testCase.0,
              message: "rotated values or preserved suffix differ from the independent reference"))
        }
      }
      let tensor = try FloatTensor([1, 2, 3, 4], shape: [1, 1, 4])
      passed += AttentionJudgeOracle.expectError(
        name: "reject odd rotary dimension", failures: &failures
      ) {
        _ = try implementation(tensor, tensor, 3, 10_000, 0, 0)
      }
      passed += AttentionJudgeOracle.expectError(
        name: "reject rotary dimension beyond head", failures: &failures
      ) {
        _ = try implementation(tensor, tensor, 6, 10_000, 0, 0)
      }
      passed += AttentionJudgeOracle.expectError(
        name: "reject negative position offset", failures: &failures
      ) {
        _ = try implementation(tensor, tensor, 4, 10_000, -1, 0)
      }
    } catch {
      failures.append(
        JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 5, failures: failures)
  }

  private static func reference(
    queries: FloatTensor,
    keys: FloatTensor,
    rotaryDimension: Int,
    base: Float,
    queryOffset: Int,
    keyOffset: Int
  ) throws -> RoPEResult {
    func rotate(_ tensor: FloatTensor, positionOffset: Int) throws -> FloatTensor {
      var output = tensor.storage
      let heads = tensor.shape[1]
      let headDimension = tensor.shape[2]
      for sequence in 0..<tensor.shape[0] {
        let position = Double(positionOffset + sequence)
        for head in 0..<heads {
          let headStart = (sequence * heads + head) * headDimension
          for pairStart in stride(from: 0, to: rotaryDimension, by: 2) {
            let pair = pairStart / 2
            let angle = position / pow(Double(base), Double(2 * pair) / Double(rotaryDimension))
            let first = Double(tensor.storage[headStart + pairStart])
            let second = Double(tensor.storage[headStart + pairStart + 1])
            output[headStart + pairStart] = Float(first * cos(angle) - second * sin(angle))
            output[headStart + pairStart + 1] = Float(first * sin(angle) + second * cos(angle))
          }
        }
      }
      return try FloatTensor(output, shape: tensor.shape)
    }
    return RoPEResult(
      queries: try rotate(queries, positionOffset: queryOffset),
      keys: try rotate(keys, positionOffset: keyOffset)
    )
  }

  private static func equal(_ lhs: FloatTensor, _ rhs: FloatTensor) -> Bool {
    lhs.shape == rhs.shape
      && zip(lhs.storage, rhs.storage).allSatisfy { actual, expected in
        abs(actual - expected) <= absoluteTolerance + relativeTolerance * abs(expected)
      }
  }
}
