import InferenceSchoolCore

public enum P002TensorStorageExercise {
    public static func gather(
        storage: [Float],
        shape: [Int],
        indices: [[Int]]
    ) throws -> [Float] {
        _ = try FloatTensor(storage, shape: shape)
        for index in indices {
            _ = try TensorLayout.rowMajor(shape: shape).offset(for: index)
        }

        // TODO: Read each logical index through the checked tensor view.
        return Array(repeating: 0, count: indices.count)
    }
}