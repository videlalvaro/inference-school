import Foundation

public enum SoftmaxError: Error, Equatable, LocalizedError {
    case emptyRow
    case nonFiniteInput(row: Int, column: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyRow: "Softmax rows must contain at least one value."
        case let .nonFiniteInput(row, column): "Softmax input at [\(row), \(column)] must be finite."
        }
    }
}

public typealias SoftmaxImplementation = (_ logits: FloatTensor) throws -> FloatTensor

public enum P009SoftmaxJudge {
    public static let probabilityTolerance: Float = 3e-5

    private struct ValueCase {
        let name: String
        let logits: FloatTensor
    }

    public static func evaluate(_ implementation: SoftmaxImplementation) -> JudgeReport {
        let cases: [ValueCase]
        do {
            cases = [
                ValueCase(name: "ordinary rows", logits: try FloatTensor([1, 2, 3, -1, 0, 1], shape: [2, 3])),
                ValueCase(name: "large positive logits", logits: try FloatTensor([10_000, 10_001, 9_999], shape: [1, 3])),
                ValueCase(name: "all-negative large logits", logits: try FloatTensor([-10_000, -10_001, -9_999], shape: [1, 3])),
                ValueCase(name: "single-value rows", logits: try FloatTensor([4, -7], shape: [2, 1])),
                ValueCase(name: "crosses threadgroup width", logits: try FloatTensor((0..<257).map { Float(($0 % 29) - 14) }, shape: [1, 257])),
                ValueCase(name: "zero rows", logits: try FloatTensor([], shape: [0, 4])),
            ]
        } catch {
            return JudgeReport(passedCaseCount: 0, totalCaseCount: 9, failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)])
        }
        var failures: [JudgeFailure] = []
        var passed = 0
        for testCase in cases {
            do {
                let actual = try implementation(testCase.logits)
                let expected = reference(testCase.logits)
                let sumsAreOne = actual.shape[0] == 0 || (0..<actual.shape[0]).allSatisfy { row in
                    let start = row * actual.shape[1]
                    return abs(actual.storage[start..<(start + actual.shape[1])].reduce(0, +) - 1) <= probabilityTolerance
                }
                if actual.shape == testCase.logits.shape,
                   approximatelyEqual(actual.storage, expected), sumsAreOne {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(caseName: testCase.name, message: "expected probabilities \(expected), received shape \(actual.shape) and values \(actual.storage)"))
                }
            } catch {
                failures.append(JudgeFailure(caseName: testCase.name, message: "unexpected error: \(error.localizedDescription)"))
            }
        }
        passed += expectError(name: "reject rank-one logits", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [2]))
        }
        passed += expectError(name: "reject empty rows", failures: &failures) {
            _ = try implementation(FloatTensor([], shape: [2, 0]))
        }
        passed += expectError(name: "reject non-finite logits", failures: &failures) {
            _ = try implementation(FloatTensor([1, .infinity], shape: [1, 2]))
        }
        return JudgeReport(passedCaseCount: passed, totalCaseCount: cases.count + 3, failures: failures)
    }

    private static func reference(_ logits: FloatTensor) -> [Float] {
        let rows = logits.shape[0]
        let columns = logits.shape[1]
        return (0..<rows).flatMap { row -> [Float] in
            let values = (0..<columns).map { Double(logits.storage[row * columns + $0]) }
            let maximum = values.max()!
            let exponentials = values.map { exp($0 - maximum) }
            let sum = exponentials.reduce(0, +)
            return exponentials.map { Float($0 / sum) }
        }
    }

    private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { abs($0 - $1) <= probabilityTolerance }
    }

    private static func expectError(name: String, failures: inout [JudgeFailure], operation: () throws -> Void) -> Int {
        do {
            try operation()
            failures.append(JudgeFailure(caseName: name, message: "expected an error, but the implementation returned"))
            return 0
        } catch { return 1 }
    }
}