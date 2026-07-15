import Foundation

public enum VectorDotError: Error, Equatable, LocalizedError {
    case lengthMismatch(lhs: Int, rhs: Int)

    public var errorDescription: String? {
        switch self {
        case let .lengthMismatch(lhs, rhs):
            "Vector lengths must match; received \(lhs) and \(rhs)."
        }
    }
}

public typealias VectorDotImplementation = (
    _ lhs: [Float],
    _ rhs: [Float]
) throws -> Float

public struct JudgeFailure: Sendable {
    public let caseName: String
    public let message: String

    public init(caseName: String, message: String) {
        self.caseName = caseName
        self.message = message
    }
}

public struct JudgeReport: Sendable {
    public let passedCaseCount: Int
    public let totalCaseCount: Int
    public let failures: [JudgeFailure]

    public var isPassing: Bool {
        failures.isEmpty && passedCaseCount == totalCaseCount
    }

    public init(
        passedCaseCount: Int,
        totalCaseCount: Int,
        failures: [JudgeFailure]
    ) {
        self.passedCaseCount = passedCaseCount
        self.totalCaseCount = totalCaseCount
        self.failures = failures
    }
}

public enum P001VectorDotJudge {
    private struct ValueCase: Sendable {
        let name: String
        let lhs: [Float]
        let rhs: [Float]
    }

    public static func evaluate(_ implementation: VectorDotImplementation) -> JudgeReport {
        let valueCases = makeValueCases()
        var failures: [JudgeFailure] = []
        var passed = 0

        for testCase in valueCases {
            do {
                let actual = try implementation(testCase.lhs, testCase.rhs)
                let expected = reference(testCase.lhs, testCase.rhs)

                if approximatelyEqual(actual, expected) {
                    passed += 1
                } else {
                    failures.append(
                        JudgeFailure(
                            caseName: testCase.name,
                            message: "expected \(expected), received \(actual)"
                        )
                    )
                }
            } catch {
                failures.append(
                    JudgeFailure(
                        caseName: testCase.name,
                        message: "unexpected error: \(error.localizedDescription)"
                    )
                )
            }
        }

        let mismatchCaseName = "reject mismatched lengths"
        do {
            _ = try implementation([1, 2], [1])
            failures.append(
                JudgeFailure(
                    caseName: mismatchCaseName,
                    message: "expected an error, but the implementation returned a value"
                )
            )
        } catch {
            passed += 1
        }

        return JudgeReport(
            passedCaseCount: passed,
            totalCaseCount: valueCases.count + 1,
            failures: failures
        )
    }

    private static func makeValueCases() -> [ValueCase] {
        let longLHS = (0..<1_025).map { index in
            Float((index % 17) - 8) / 8
        }
        let longRHS = (0..<1_025).map { index in
            Float((index % 11) - 5) / 5
        }

        return [
            ValueCase(name: "empty vectors", lhs: [], rhs: []),
            ValueCase(name: "single element", lhs: [3], rhs: [-2]),
            ValueCase(name: "mixed signs", lhs: [1, -2, 3, -4], rhs: [0.5, 2, -1, -0.25]),
            ValueCase(name: "crosses threadgroup boundaries", lhs: longLHS, rhs: longRHS),
        ]
    }

    private static func reference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        Float(zip(lhs, rhs).reduce(0.0) { sum, pair in
            sum + Double(pair.0) * Double(pair.1)
        })
    }

    private static func approximatelyEqual(_ lhs: Float, _ rhs: Float) -> Bool {
        let scale = max(1, abs(lhs), abs(rhs))
        return abs(lhs - rhs) <= 1e-5 * scale
    }
}