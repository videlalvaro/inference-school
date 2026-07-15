import Foundation

public typealias MatrixTransposeImplementation = (_ input: FloatTensor) throws -> FloatTensor

public enum P003TransposeJudge {
    private struct ValueCase {
        let name: String
        let input: FloatTensor
        let expectedStorage: [Float]
        let expectedShape: [Int]
    }

    public static func evaluate(_ implementation: MatrixTransposeImplementation) -> JudgeReport {
        let valueCases: [ValueCase]
        do {
            valueCases = [
                ValueCase(
                    name: "rectangular matrix",
                    input: try FloatTensor([1, 2, 3, 4, 5, 6], shape: [2, 3]),
                    expectedStorage: [1, 4, 2, 5, 3, 6],
                    expectedShape: [3, 2]
                ),
                ValueCase(
                    name: "single row",
                    input: try FloatTensor([-2, 0, 7], shape: [1, 3]),
                    expectedStorage: [-2, 0, 7],
                    expectedShape: [3, 1]
                ),
                ValueCase(
                    name: "empty rows",
                    input: try FloatTensor([], shape: [0, 5]),
                    expectedStorage: [],
                    expectedShape: [5, 0]
                ),
                ValueCase(
                    name: "crosses tile edges",
                    input: try FloatTensor(
                        (0..<(17 * 19)).map { Float($0 - 100) / 7 },
                        shape: [17, 19]
                    ),
                    expectedStorage: transposeReference(
                        (0..<(17 * 19)).map { Float($0 - 100) / 7 },
                        rows: 17,
                        columns: 19
                    ),
                    expectedShape: [19, 17]
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
                let actual = try implementation(testCase.input)
                if actual.shape == testCase.expectedShape,
                   approximatelyEqual(actual.storage, testCase.expectedStorage) {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(
                        caseName: testCase.name,
                        message: "expected shape \(testCase.expectedShape) and values \(testCase.expectedStorage.prefix(8)); received shape \(actual.shape) and values \(actual.storage.prefix(8))"
                    ))
                }
            } catch {
                failures.append(JudgeFailure(
                    caseName: testCase.name,
                    message: "unexpected error: \(error.localizedDescription)"
                ))
            }
        }

        let rankCase = "reject non-matrix input"
        do {
            _ = try implementation(FloatTensor([1, 2], shape: [2]))
            failures.append(JudgeFailure(
                caseName: rankCase,
                message: "expected a rank error, but the implementation returned"
            ))
        } catch {
            passed += 1
        }

        return JudgeReport(
            passedCaseCount: passed,
            totalCaseCount: valueCases.count + 1,
            failures: failures
        )
    }

    private static func transposeReference(
        _ input: [Float],
        rows: Int,
        columns: Int
    ) -> [Float] {
        var output = Array(repeating: Float.zero, count: input.count)
        for row in 0..<rows {
            for column in 0..<columns {
                output[column * rows + row] = input[row * columns + column]
            }
        }
        return output
    }

    private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return zip(lhs, rhs).allSatisfy { left, right in
            let scale = max(1, abs(left), abs(right))
            return abs(left - right) <= 1e-6 * scale
        }
    }
}