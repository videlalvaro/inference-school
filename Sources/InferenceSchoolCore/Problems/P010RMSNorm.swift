import Foundation

public enum RMSNormError: Error, Equatable, LocalizedError {
    case emptyFeatureWidth
    case gammaWidthMismatch(expected: Int, actual: Int)
    case invalidEpsilon(Float)
    case nonFiniteInput(row: Int, column: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyFeatureWidth: "RMSNorm rows must contain at least one feature."
        case let .gammaWidthMismatch(expected, actual): "RMSNorm gamma width must be \(expected); received \(actual)."
        case let .invalidEpsilon(value): "RMSNorm epsilon must be finite and positive; received \(value)."
        case let .nonFiniteInput(row, column): "RMSNorm input at [\(row), \(column)] must be finite."
        }
    }
}

public typealias RMSNormImplementation = (
    _ input: FloatTensor,
    _ gamma: FloatTensor,
    _ epsilon: Float
) throws -> FloatTensor

public enum P010RMSNormJudge {
    public static let absoluteTolerance: Float = 4e-5
    public static let relativeTolerance: Float = 5e-5

    private struct ValueCase {
        let name: String
        let input: FloatTensor
        let gamma: FloatTensor
        let epsilon: Float
    }

    public static func evaluate(_ implementation: RMSNormImplementation) -> JudgeReport {
        let cases: [ValueCase]
        do {
            cases = [
                ValueCase(name: "mixed rows and scale", input: try FloatTensor([1, 2, -3, 4, -2, 1], shape: [2, 3]), gamma: try FloatTensor([1, 0.5, 2], shape: [3]), epsilon: 1e-5),
                ValueCase(name: "constant row", input: try FloatTensor([4, 4, 4, 4], shape: [1, 4]), gamma: try FloatTensor([1, 1, 1, 1], shape: [4]), epsilon: 1e-6),
                ValueCase(name: "epsilon controls zero row", input: try FloatTensor([0, 0, 0], shape: [1, 3]), gamma: try FloatTensor([2, 3, 4], shape: [3]), epsilon: 0.25),
                ValueCase(name: "large finite values", input: try FloatTensor([1e18, -1e18, 5e17], shape: [1, 3]), gamma: try FloatTensor([1, 2, 0.5], shape: [3]), epsilon: 1e-5),
                ValueCase(name: "zero rows", input: try FloatTensor([], shape: [0, 3]), gamma: try FloatTensor([1, 2, 3], shape: [3]), epsilon: 1e-5),
            ]
        } catch {
            return JudgeReport(passedCaseCount: 0, totalCaseCount: 9, failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)])
        }
        var failures: [JudgeFailure] = []
        var passed = 0
        for testCase in cases {
            do {
                let actual = try implementation(testCase.input, testCase.gamma, testCase.epsilon)
                let expected = reference(input: testCase.input, gamma: testCase.gamma, epsilon: testCase.epsilon)
                if actual.shape == testCase.input.shape, approximatelyEqual(actual.storage, expected) {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(caseName: testCase.name, message: "expected \(expected), received shape \(actual.shape) and values \(actual.storage)"))
                }
            } catch {
                failures.append(JudgeFailure(caseName: testCase.name, message: "unexpected error: \(error.localizedDescription)"))
            }
        }
        passed += expectError(name: "reject input rank", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [2]), FloatTensor([1, 1], shape: [2]), 1e-5)
        }
        passed += expectError(name: "reject gamma width", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [1, 2]), FloatTensor([1], shape: [1]), 1e-5)
        }
        passed += expectError(name: "reject nonpositive epsilon", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [1, 2]), FloatTensor([1, 1], shape: [2]), 0)
        }
        passed += expectError(name: "reject empty feature width", failures: &failures) {
            _ = try implementation(FloatTensor([], shape: [2, 0]), FloatTensor([], shape: [0]), 1e-5)
        }
        return JudgeReport(passedCaseCount: passed, totalCaseCount: cases.count + 4, failures: failures)
    }

    private static func reference(input: FloatTensor, gamma: FloatTensor, epsilon: Float) -> [Float] {
        let rows = input.shape[0]
        let width = input.shape[1]
        return (0..<rows).flatMap { row -> [Float] in
            let meanSquare = (0..<width).reduce(0.0) { sum, column in
                let value = Double(input.storage[row * width + column])
                return sum + value * value
            } / Double(width)
            let inverseRMS = 1 / sqrt(meanSquare + Double(epsilon))
            return (0..<width).map { column in
                Float(Double(input.storage[row * width + column]) * inverseRMS * Double(gamma.storage[column]))
            }
        }
    }

    private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        lhs.count == rhs.count && zip(lhs, rhs).allSatisfy {
            abs($0 - $1) <= absoluteTolerance + relativeTolerance * abs($1)
        }
    }

    private static func expectError(name: String, failures: inout [JudgeFailure], operation: () throws -> Void) -> Int {
        do {
            try operation()
            failures.append(JudgeFailure(caseName: name, message: "expected an error, but the implementation returned"))
            return 0
        } catch { return 1 }
    }
}