import Foundation

public enum ResidualPrecisionPolicy: Sendable, Equatable {
    case float32
    case float16AfterEachAdd
}

public enum ResidualStreamError: Error, Equatable, LocalizedError {
    case updateShapeMismatch(index: Int, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case let .updateShapeMismatch(index, expected, actual):
            "Residual update \(index) must have shape \(expected); received \(actual)."
        }
    }
}

public struct ResidualPrecisionComparison: Sendable, Equatable {
    public let float32: FloatTensor
    public let float16AfterEachAdd: FloatTensor
    public let maximumAbsoluteDifference: Float

    public init(float32: FloatTensor, float16AfterEachAdd: FloatTensor) {
        self.float32 = float32
        self.float16AfterEachAdd = float16AfterEachAdd
        self.maximumAbsoluteDifference = zip(float32.storage, float16AfterEachAdd.storage)
            .map { abs($0 - $1) }
            .max() ?? 0
    }
}

public typealias ResidualAccumulationImplementation = (
    _ initial: FloatTensor,
    _ updates: [FloatTensor],
    _ policy: ResidualPrecisionPolicy
) throws -> FloatTensor

public enum P011ResidualPrecisionJudge {
    private struct ValueCase {
        let name: String
        let initial: FloatTensor
        let updates: [FloatTensor]
        let policy: ResidualPrecisionPolicy
    }

    public static func evaluate(_ implementation: ResidualAccumulationImplementation) -> JudgeReport {
        let cases: [ValueCase]
        do {
            let largeInitial = try FloatTensor([4096, -4096], shape: [2])
            let smallUpdates = try (0..<4).map { _ in try FloatTensor([0.5, -0.5], shape: [2]) }
            cases = [
                ValueCase(name: "Float32 keeps small updates", initial: largeInitial, updates: smallUpdates, policy: .float32),
                ValueCase(name: "Float16 downcast loses sub-ULP updates", initial: largeInitial, updates: smallUpdates, policy: .float16AfterEachAdd),
                ValueCase(name: "matrix residual shape", initial: try FloatTensor([1, 2, 3, 4], shape: [2, 2]), updates: [try FloatTensor([0.25, -0.5, 1, -2], shape: [2, 2])], policy: .float32),
                ValueCase(name: "no updates", initial: try FloatTensor([1.25, -2.5], shape: [2]), updates: [], policy: .float16AfterEachAdd),
            ]
        } catch {
            return JudgeReport(passedCaseCount: 0, totalCaseCount: 5, failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)])
        }
        var failures: [JudgeFailure] = []
        var passed = 0
        for testCase in cases {
            do {
                let actual = try implementation(testCase.initial, testCase.updates, testCase.policy)
                let expected = reference(initial: testCase.initial, updates: testCase.updates, policy: testCase.policy)
                if actual.shape == testCase.initial.shape, actual.storage == expected {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(caseName: testCase.name, message: "expected \(expected), received shape \(actual.shape) and values \(actual.storage)"))
                }
            } catch {
                failures.append(JudgeFailure(caseName: testCase.name, message: "unexpected error: \(error.localizedDescription)"))
            }
        }
        do {
            _ = try implementation(
                FloatTensor([1, 2], shape: [2]),
                [FloatTensor([1, 2], shape: [1, 2])],
                .float32
            )
            failures.append(JudgeFailure(caseName: "reject update shape", message: "expected an error, but the implementation returned"))
        } catch {
            passed += 1
        }
        return JudgeReport(passedCaseCount: passed, totalCaseCount: cases.count + 1, failures: failures)
    }

    private static func reference(initial: FloatTensor, updates: [FloatTensor], policy: ResidualPrecisionPolicy) -> [Float] {
        var values = initial.storage
        for update in updates {
            for index in values.indices {
                let sum = values[index] + update.storage[index]
                values[index] = policy == .float32 ? sum : Float(Float16(sum))
            }
        }
        return values
    }
}