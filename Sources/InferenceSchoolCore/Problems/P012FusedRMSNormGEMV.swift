import Foundation

public enum FusedRMSNormGEMVError: Error, Equatable, LocalizedError {
    case inputWidthMismatch(expected: Int, actual: Int)
    case gammaWidthMismatch(expected: Int, actual: Int)
    case emptyInput
    case invalidEpsilon(Float)

    public var errorDescription: String? {
        switch self {
        case let .inputWidthMismatch(expected, actual): "Projection input width must be \(expected); received \(actual)."
        case let .gammaWidthMismatch(expected, actual): "Gamma width must be \(expected); received \(actual)."
        case .emptyInput: "Fused RMSNorm plus projection requires at least one input feature."
        case let .invalidEpsilon(value): "Epsilon must be finite and positive; received \(value)."
        }
    }
}

public typealias FusedRMSNormGEMVImplementation = (
    _ input: FloatTensor,
    _ gamma: FloatTensor,
    _ weights: FloatTensor,
    _ epsilon: Float
) throws -> FloatTensor

public enum P012FusedRMSNormGEMVJudge {
    public static let relativeTolerance: Float = 8e-5

    public static func evaluate(_ implementation: FusedRMSNormGEMVImplementation) -> JudgeReport {
        let cases: [(String, FloatTensor, FloatTensor, FloatTensor, Float)]
        do {
            let wideInput = (0..<257).map { Float(($0 % 23) - 11) / 7 }
            let wideGamma = (0..<257).map { 0.5 + Float($0 % 9) / 10 }
            let wideWeights = (0..<(3 * 257)).map { Float(($0 % 31) - 15) / 19 }
            cases = [
                ("small fused projection", try FloatTensor([1, -2, 3], shape: [3]), try FloatTensor([1, 0.5, 2], shape: [3]), try FloatTensor([1, 2, 0, -1, 0.5, 3], shape: [2, 3]), 1e-5),
                ("crosses reduction boundary", try FloatTensor(wideInput, shape: [257]), try FloatTensor(wideGamma, shape: [257]), try FloatTensor(wideWeights, shape: [3, 257]), 1e-6),
                ("zero output rows", try FloatTensor([1, 2, 3, 4], shape: [4]), try FloatTensor([1, 1, 1, 1], shape: [4]), try FloatTensor([], shape: [0, 4]), 1e-5),
            ]
        } catch {
            return JudgeReport(passedCaseCount: 0, totalCaseCount: 7, failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)])
        }
        var failures: [JudgeFailure] = []
        var passed = 0
        for (name, input, gamma, weights, epsilon) in cases {
            do {
                let actual = try implementation(input, gamma, weights, epsilon)
                let expected = reference(input: input, gamma: gamma, weights: weights, epsilon: epsilon)
                if actual.shape == [weights.shape[0]], approximatelyEqual(actual.storage, expected) {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(caseName: name, message: "expected \(expected), received shape \(actual.shape) and values \(actual.storage)"))
                }
            } catch {
                failures.append(JudgeFailure(caseName: name, message: "unexpected error: \(error.localizedDescription)"))
            }
        }
        passed += expectError(name: "reject input rank", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [1, 2]), FloatTensor([1, 1], shape: [2]), FloatTensor([1, 1], shape: [1, 2]), 1e-5)
        }
        passed += expectError(name: "reject input width", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [2]), FloatTensor([1, 1], shape: [2]), FloatTensor([1, 2, 3], shape: [1, 3]), 1e-5)
        }
        passed += expectError(name: "reject gamma width", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [2]), FloatTensor([1], shape: [1]), FloatTensor([1, 2], shape: [1, 2]), 1e-5)
        }
        passed += expectError(name: "reject epsilon", failures: &failures) {
            _ = try implementation(FloatTensor([1, 2], shape: [2]), FloatTensor([1, 1], shape: [2]), FloatTensor([1, 2], shape: [1, 2]), -1)
        }
        return JudgeReport(passedCaseCount: passed, totalCaseCount: cases.count + 4, failures: failures)
    }

    private static func reference(input: FloatTensor, gamma: FloatTensor, weights: FloatTensor, epsilon: Float) -> [Float] {
        let width = input.shape[0]
        let meanSquare = input.storage.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(width)
        let inverseRMS = 1 / sqrt(meanSquare + Double(epsilon))
        return (0..<weights.shape[0]).map { row in
            Float((0..<width).reduce(0.0) { sum, column in
                let normalized = Double(input.storage[column]) * inverseRMS * Double(gamma.storage[column])
                return sum + Double(weights.storage[row * width + column]) * normalized
            })
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
        } catch { return 1 }
    }
}