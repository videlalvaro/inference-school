import Foundation

public typealias GEMMImplementation = (
    _ lhs: FloatTensor,
    _ rhs: FloatTensor
) throws -> FloatTensor

public enum P005GEMMJudge {
    private struct ValueCase {
        let name: String
        let lhs: FloatTensor
        let rhs: FloatTensor
        let expected: [Float]
        let expectedShape: [Int]
    }

    public static func evaluate(_ implementation: GEMMImplementation) -> JudgeReport {
        let valueCases: [ValueCase]
        do {
            let edgeLHS = (0..<(17 * 19)).map { Float(($0 % 23) - 11) / 12 }
            let edgeRHS = (0..<(19 * 18)).map { Float(($0 % 31) - 15) / 16 }
            valueCases = [
                ValueCase(
                    name: "small rectangular product",
                    lhs: try FloatTensor([1, 2, 3, 4, 5, 6], shape: [2, 3]),
                    rhs: try FloatTensor([1, 2, 0, -1, 3, 0], shape: [3, 2]),
                    expected: [10, 0, 22, 3],
                    expectedShape: [2, 2]
                ),
                ValueCase(
                    name: "zero inner dimension",
                    lhs: try FloatTensor([], shape: [2, 0]),
                    rhs: try FloatTensor([], shape: [0, 3]),
                    expected: Array(repeating: 0, count: 6),
                    expectedShape: [2, 3]
                ),
                ValueCase(
                    name: "zero output rows",
                    lhs: try FloatTensor([], shape: [0, 4]),
                    rhs: try FloatTensor((0..<12).map(Float.init), shape: [4, 3]),
                    expected: [],
                    expectedShape: [0, 3]
                ),
                ValueCase(
                    name: "partial tiles in every dimension",
                    lhs: try FloatTensor(edgeLHS, shape: [17, 19]),
                    rhs: try FloatTensor(edgeRHS, shape: [19, 18]),
                    expected: gemmReference(edgeLHS, edgeRHS, m: 17, k: 19, n: 18),
                    expectedShape: [17, 18]
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
                let actual = try implementation(testCase.lhs, testCase.rhs)
                if actual.shape == testCase.expectedShape,
                   approximatelyEqual(actual.storage, testCase.expected) {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(
                        caseName: testCase.name,
                        message: "expected shape \(testCase.expectedShape) and values \(testCase.expected.prefix(8)); received shape \(actual.shape) and values \(actual.storage.prefix(8))"
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
                FloatTensor([1, 2, 3, 4], shape: [2, 2])
            )
            failures.append(JudgeFailure(
                caseName: "reject operand rank",
                message: "expected a rank error, but the implementation returned"
            ))
        } catch {
            passed += 1
        }
        do {
            _ = try implementation(
                FloatTensor([1, 2, 3, 4], shape: [2, 2]),
                FloatTensor([1, 2, 3], shape: [3, 1])
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

    private static func gemmReference(
        _ lhs: [Float],
        _ rhs: [Float],
        m: Int,
        k: Int,
        n: Int
    ) -> [Float] {
        var output = Array(repeating: Float.zero, count: m * n)
        for row in 0..<m {
            for column in 0..<n {
                let sum: Double = (0..<k).reduce(0.0) { accumulator, inner in
                    accumulator + Double(lhs[row * k + inner]) * Double(rhs[inner * n + column])
                }
                output[row * n + column] = Float(sum)
            }
        }
        return output
    }

    private static func approximatelyEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return zip(lhs, rhs).allSatisfy { left, right in
            let scale = max(1, abs(left), abs(right))
            return abs(left - right) <= 4e-5 * scale
        }
    }
}