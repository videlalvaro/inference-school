import Foundation

public enum SwiGLUError: Error, Equatable, LocalizedError {
    case hiddenProjectionShapeMismatch(gate: [Int], up: [Int])
    case inputWidthMismatch(expected: Int, actual: Int)
    case downWidthMismatch(expected: Int, actual: Int)
    case gateValueShapeMismatch(gate: [Int], up: [Int])

    public var errorDescription: String? {
        switch self {
        case let .hiddenProjectionShapeMismatch(gate, up):
            "Gate and up projection shapes must match; received \(gate) and \(up)."
        case let .inputWidthMismatch(expected, actual):
            "Projection input width must be \(expected); received \(actual)."
        case let .downWidthMismatch(expected, actual):
            "Down projection width must be \(expected); received \(actual)."
        case let .gateValueShapeMismatch(gate, up):
            "Gate and up values must have the same shape; received \(gate) and \(up)."
        }
    }
}

public typealias SwiGLUImplementation = (
    _ input: FloatTensor,
    _ gateWeights: FloatTensor,
    _ upWeights: FloatTensor,
    _ downWeights: FloatTensor
) throws -> FloatTensor

public typealias SwiGLUGateImplementation = (
    _ gate: FloatTensor,
    _ up: FloatTensor
) throws -> FloatTensor

public enum P008SwiGLUJudge {
    public static let relativeTolerance: Float = 4e-5

    private struct ValueCase {
        let name: String
        let input: FloatTensor
        let gate: FloatTensor
        let up: FloatTensor
        let down: FloatTensor
    }

    public static func evaluate(_ implementation: SwiGLUImplementation) -> JudgeReport {
        let cases: [ValueCase]
        do {
            cases = [
                ValueCase(
                    name: "three projections with distinct output width",
                    input: try FloatTensor([1, -2], shape: [2]),
                    gate: try FloatTensor([1, 0, 0, 1, 1, -1], shape: [3, 2]),
                    up: try FloatTensor([2, 1, -1, 1, 0.5, 2], shape: [3, 2]),
                    down: try FloatTensor([1, 0.5, -1, -0.25, 2, 0.75], shape: [2, 3])
                ),
                ValueCase(
                    name: "zero activations still preserve output shape",
                    input: try FloatTensor([0, 0, 0], shape: [3]),
                    gate: try FloatTensor(Array(repeating: 1, count: 12), shape: [4, 3]),
                    up: try FloatTensor(Array(repeating: -2, count: 12), shape: [4, 3]),
                    down: try FloatTensor(Array(repeating: 0.5, count: 8), shape: [2, 4])
                ),
                ValueCase(
                    name: "empty hidden width",
                    input: try FloatTensor([1, 2], shape: [2]),
                    gate: try FloatTensor([], shape: [0, 2]),
                    up: try FloatTensor([], shape: [0, 2]),
                    down: try FloatTensor([], shape: [3, 0])
                ),
            ]
        } catch {
            return JudgeReport(passedCaseCount: 0, totalCaseCount: 6, failures: [
                JudgeFailure(caseName: "judge setup", message: error.localizedDescription),
            ])
        }

        var passed = 0
        var failures: [JudgeFailure] = []
        for testCase in cases {
            do {
                let actual = try implementation(testCase.input, testCase.gate, testCase.up, testCase.down)
                let expected = reference(
                    input: testCase.input,
                    gate: testCase.gate,
                    up: testCase.up,
                    down: testCase.down
                )
                if actual.shape == [testCase.down.shape[0]], approximatelyEqual(actual.storage, expected) {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(
                        caseName: testCase.name,
                        message: "expected shape \([testCase.down.shape[0]]) and values \(expected); received shape \(actual.shape) and values \(actual.storage)"
                    ))
                }
            } catch {
                failures.append(JudgeFailure(caseName: testCase.name, message: "unexpected error: \(error.localizedDescription)"))
            }
        }

        passed += expectError(name: "reject input rank", failures: &failures) {
            _ = try implementation(
                FloatTensor([1, 2], shape: [1, 2]),
                FloatTensor([1, 2], shape: [1, 2]),
                FloatTensor([1, 2], shape: [1, 2]),
                FloatTensor([1], shape: [1, 1])
            )
        }
        passed += expectError(name: "reject gate/up mismatch", failures: &failures) {
            _ = try implementation(
                FloatTensor([1, 2], shape: [2]),
                FloatTensor([1, 2], shape: [1, 2]),
                FloatTensor([1, 2, 3, 4], shape: [2, 2]),
                FloatTensor([1], shape: [1, 1])
            )
        }
        passed += expectError(name: "reject down hidden width", failures: &failures) {
            _ = try implementation(
                FloatTensor([1, 2], shape: [2]),
                FloatTensor([1, 2], shape: [1, 2]),
                FloatTensor([1, 2], shape: [1, 2]),
                FloatTensor([1, 2], shape: [1, 2])
            )
        }
        return JudgeReport(passedCaseCount: passed, totalCaseCount: cases.count + 3, failures: failures)
    }

    public static func evaluateGate(_ implementation: SwiGLUGateImplementation) -> JudgeReport {
        let gate: FloatTensor
        let up: FloatTensor
        do {
            gate = try FloatTensor([-20, -2, 0, 2, 20, 1], shape: [2, 3])
            up = try FloatTensor([3, -1, 4, 0.5, -2, 8], shape: [2, 3])
        } catch {
            return JudgeReport(passedCaseCount: 0, totalCaseCount: 2, failures: [
                JudgeFailure(caseName: "judge setup", message: error.localizedDescription),
            ])
        }
        var failures: [JudgeFailure] = []
        var passed = 0
        do {
            let actual = try implementation(gate, up)
            let expected = zip(gate.storage, up.storage).map { gateValue, upValue in
                let x = Double(gateValue)
                return Float((x / (1 + exp(-x))) * Double(upValue))
            }
            if actual.shape == gate.shape, approximatelyEqual(actual.storage, expected) {
                passed += 1
            } else {
                failures.append(JudgeFailure(caseName: "fused SiLU gate", message: "expected \(expected), received \(actual.storage)"))
            }
        } catch {
            failures.append(JudgeFailure(caseName: "fused SiLU gate", message: "unexpected error: \(error.localizedDescription)"))
        }
        passed += expectError(name: "reject gate value shape mismatch", failures: &failures) {
            _ = try implementation(gate, FloatTensor(Array(repeating: 1, count: 6), shape: [6]))
        }
        return JudgeReport(passedCaseCount: passed, totalCaseCount: 2, failures: failures)
    }

    private static func reference(input: FloatTensor, gate: FloatTensor, up: FloatTensor, down: FloatTensor) -> [Float] {
        let hidden = gate.shape[0]
        let width = input.shape[0]
        let gated = (0..<hidden).map { row -> Double in
            var gateValue = 0.0
            var upValue = 0.0
            for column in 0..<width {
                gateValue += Double(gate.storage[row * width + column]) * Double(input.storage[column])
                upValue += Double(up.storage[row * width + column]) * Double(input.storage[column])
            }
            return (gateValue / (1 + exp(-gateValue))) * upValue
        }
        return (0..<down.shape[0]).map { row in
            Float((0..<hidden).reduce(0.0) { $0 + Double(down.storage[row * hidden + $1]) * gated[$1] })
        }
    }

    private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        lhs.count == rhs.count && zip(lhs, rhs).allSatisfy {
            abs($0 - $1) <= relativeTolerance * max(1, abs($0), abs($1))
        }
    }

    private static func expectError(name: String, failures: inout [JudgeFailure], operation: () throws -> Void) -> Int {
        do {
            try operation()
            failures.append(JudgeFailure(caseName: name, message: "expected an error, but the implementation returned"))
            return 0
        } catch {
            return 1
        }
    }
}