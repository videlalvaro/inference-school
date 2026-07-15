import Foundation

public enum DenseLinearAlgebraError: Error, Equatable, LocalizedError {
    case innerDimensionMismatch(operation: String, lhs: Int, rhs: Int)

    public var errorDescription: String? {
        switch self {
        case let .innerDimensionMismatch(operation, lhs, rhs):
            "\(operation) inner dimensions must match; received \(lhs) and \(rhs)."
        }
    }
}

public typealias GEMVImplementation = (
    _ matrix: FloatTensor,
    _ vector: FloatTensor
) throws -> FloatTensor

public enum P004GEMVJudge {
    private struct ValueCase {
        let name: String
        let matrix: FloatTensor
        let vector: FloatTensor
        let expected: [Float]
    }

    public static func evaluate(_ implementation: GEMVImplementation) -> JudgeReport {
        let valueCases: [ValueCase]
        do {
            let wideMatrix = (0..<(3 * 257)).map { Float(($0 % 29) - 14) / 13 }
            let wideVector = (0..<257).map { Float(($0 % 17) - 8) / 9 }
            valueCases = [
                ValueCase(
                    name: "small projection",
                    matrix: try FloatTensor([1, 2, 3, -1, 0.5, 4], shape: [2, 3]),
                    vector: try FloatTensor([2, -1, 0.5], shape: [3]),
                    expected: [1.5, -0.5]
                ),
                ValueCase(
                    name: "zero inner dimension",
                    matrix: try FloatTensor([], shape: [3, 0]),
                    vector: try FloatTensor([], shape: [0]),
                    expected: [0, 0, 0]
                ),
                ValueCase(
                    name: "zero output rows",
                    matrix: try FloatTensor([], shape: [0, 4]),
                    vector: try FloatTensor([1, 2, 3, 4], shape: [4]),
                    expected: []
                ),
                ValueCase(
                    name: "crosses reduction boundary",
                    matrix: try FloatTensor(wideMatrix, shape: [3, 257]),
                    vector: try FloatTensor(wideVector, shape: [257]),
                    expected: gemvReference(wideMatrix, rows: 3, columns: 257, vector: wideVector)
                ),
            ]
        } catch {
            return JudgeReport(
                passedCaseCount: 0,
                totalCaseCount: 6,
                failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)]
            )
        }

        var failures: [JudgeFailure] = []
        var passed = 0
        for testCase in valueCases {
            do {
                let actual = try implementation(testCase.matrix, testCase.vector)
                if actual.shape == [testCase.matrix.shape[0]],
                   approximatelyEqual(actual.storage, testCase.expected) {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(
                        caseName: testCase.name,
                        message: "expected \(testCase.expected), received shape \(actual.shape), values \(actual.storage)"
                    ))
                }
            } catch {
                failures.append(JudgeFailure(
                    caseName: testCase.name,
                    message: "unexpected error: \(error.localizedDescription)"
                ))
            }
        }

        do {
            _ = try implementation(
                FloatTensor([1, 2], shape: [2]),
                FloatTensor([1, 2], shape: [2])
            )
            failures.append(JudgeFailure(
                caseName: "reject matrix rank",
                message: "expected a rank error, but the implementation returned"
            ))
        } catch {
            passed += 1
        }
        do {
            _ = try implementation(
                FloatTensor([1, 2, 3, 4], shape: [2, 2]),
                FloatTensor([1, 2, 3], shape: [3])
            )
            failures.append(JudgeFailure(
                caseName: "reject inner dimension mismatch",
                message: "expected a shape error, but the implementation returned"
            ))
        } catch {
            passed += 1
        }

        return JudgeReport(
            passedCaseCount: passed,
            totalCaseCount: valueCases.count + 2,
            failures: failures
        )
    }

    private static func gemvReference(
        _ matrix: [Float],
        rows: Int,
        columns: Int,
        vector: [Float]
    ) -> [Float] {
        (0..<rows).map { row in
            Float((0..<columns).reduce(0.0) { sum, column in
                sum + Double(matrix[row * columns + column]) * Double(vector[column])
            })
        }
    }

    private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return zip(lhs, rhs).allSatisfy { left, right in
            let scale = max(1, abs(left), abs(right))
            return abs(left - right) <= 2e-5 * scale
        }
    }
}