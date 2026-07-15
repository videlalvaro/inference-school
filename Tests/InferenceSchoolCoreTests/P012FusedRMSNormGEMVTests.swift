import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P012FusedRMSNormGEMVTests: XCTestCase {
    func testCanonicalBaselinePassesJudge() {
        let report = P012FusedRMSNormGEMVJudge.evaluate(P012FusedRMSNormGEMVSolution.baseline)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsUnnormalizedProjection() {
        let report = P012FusedRMSNormGEMVJudge.evaluate { input, gamma, weights, epsilon in
            guard input.rank == 1, gamma.rank == 1, weights.rank == 2, epsilon > 0 else {
                throw FusedRMSNormGEMVError.invalidEpsilon(epsilon)
            }
            var output = Array(repeating: Float.zero, count: weights.shape[0])
            for row in output.indices {
                for column in input.storage.indices {
                    output[row] += weights.storage[row * input.storage.count + column] * input.storage[column]
                }
            }
            return try FloatTensor(output, shape: [output.count])
        }
        XCTAssertFalse(report.isPassing)
    }
}