import InferenceSchoolCore

public enum P002TensorStorageSolution {
    public static func gather(
        storage: [Float],
        shape: [Int],
        indices: [[Int]]
    ) throws -> [Float] {
        let tensor = try FloatTensor(storage, shape: shape)
        return try indices.map { try tensor.value(at: $0) }
    }
}