import Foundation

public enum Activation: UInt32, CaseIterable, Sendable {
    case relu = 0
    case geluTanhApproximation = 1
    case silu = 2

    public var name: String {
        switch self {
        case .relu: "ReLU"
        case .geluTanhApproximation: "GELU (tanh approximation)"
        case .silu: "SiLU"
        }
    }
}

public typealias ActivationImplementation = (
    _ input: FloatTensor,
    _ activation: Activation
) throws -> FloatTensor

public enum P007ActivationJudge {
    public static let absoluteTolerance: Float = 2e-6
    public static let relativeTolerance: Float = 2e-5

    private struct ValueCase {
        let name: String
        let input: FloatTensor
        let activation: Activation
    }

    public static func evaluate(_ implementation: ActivationImplementation) -> JudgeReport {
        let valueCases: [ValueCase]
        do {
            valueCases = [
                ValueCase(
                    name: "ReLU signs and zero",
                    input: try FloatTensor([-3, -0.0, 0.5, 4], shape: [4]),
                    activation: .relu
                ),
                ValueCase(
                    name: "GELU tanh approximation",
                    input: try FloatTensor([-3, -1, 0, 1, 3], shape: [5]),
                    activation: .geluTanhApproximation
                ),
                ValueCase(
                    name: "SiLU wide inputs",
                    input: try FloatTensor([-20, -2, 0, 2, 20], shape: [5]),
                    activation: .silu
                ),
                ValueCase(
                    name: "preserve matrix shape",
                    input: try FloatTensor([-2, -1, 0, 1, 2, 3], shape: [2, 3]),
                    activation: .silu
                ),
                ValueCase(
                    name: "empty tensor",
                    input: try FloatTensor([], shape: [2, 0, 3]),
                    activation: .geluTanhApproximation
                ),
            ]
        } catch {
            return JudgeReport(
                passedCaseCount: 0,
                totalCaseCount: 5,
                failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)]
            )
        }

        var failures: [JudgeFailure] = []
        var passed = 0
        for testCase in valueCases {
            do {
                let actual = try implementation(testCase.input, testCase.activation)
                let expected = testCase.input.storage.map {
                    Float(reference(Double($0), activation: testCase.activation))
                }
                if actual.shape == testCase.input.shape,
                   approximatelyEqual(actual.storage, expected) {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(
                        caseName: testCase.name,
                        message: "expected shape \(testCase.input.shape) and values \(expected); received shape \(actual.shape) and values \(actual.storage)"
                    ))
                }
            } catch {
                failures.append(JudgeFailure(
                    caseName: testCase.name,
                    message: "unexpected error: \(error.localizedDescription)"
                ))
            }
        }

        return JudgeReport(
            passedCaseCount: passed,
            totalCaseCount: valueCases.count,
            failures: failures
        )
    }

    private static func reference(_ value: Double, activation: Activation) -> Double {
        switch activation {
        case .relu:
            max(0, value)
        case .geluTanhApproximation:
            0.5 * value * (1 + tanh(sqrt(2 / Double.pi) * (value + 0.044715 * value * value * value)))
        case .silu:
            value / (1 + exp(-value))
        }
    }

    private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return zip(lhs, rhs).allSatisfy { actual, expected in
            let tolerance = absoluteTolerance + relativeTolerance * abs(expected)
            return abs(actual - expected) <= tolerance
        }
    }
}