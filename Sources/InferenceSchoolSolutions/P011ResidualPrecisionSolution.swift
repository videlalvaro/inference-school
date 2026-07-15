import InferenceSchoolCore

public enum P011ResidualPrecisionSolution {
    public static func accumulate(_ initial: FloatTensor, updates: [FloatTensor], policy: ResidualPrecisionPolicy) throws -> FloatTensor {
        for (index, update) in updates.enumerated() where update.shape != initial.shape {
            throw ResidualStreamError.updateShapeMismatch(index: index, expected: initial.shape, actual: update.shape)
        }
        var output = initial.storage
        for update in updates {
            for index in output.indices {
                let sum = output[index] + update.storage[index]
                output[index] = policy == .float32 ? sum : Float(Float16(sum))
            }
        }
        return try FloatTensor(output, shape: initial.shape)
    }

    public static func compare(_ initial: FloatTensor, updates: [FloatTensor]) throws -> ResidualPrecisionComparison {
        ResidualPrecisionComparison(
            float32: try accumulate(initial, updates: updates, policy: .float32),
            float16AfterEachAdd: try accumulate(initial, updates: updates, policy: .float16AfterEachAdd)
        )
    }
}