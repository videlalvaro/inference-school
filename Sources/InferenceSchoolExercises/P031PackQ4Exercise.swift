import InferenceSchoolCore

public enum P031PackQ4Exercise {
  public static func pack(
    _ values: [Int8],
    outputChannels: Int,
    inputChannels: Int,
    groupSize: Int,
    scales: [Float]
  ) throws -> GroupwiseQ4WeightMatrix {
    let (count, overflow) = outputChannels.multipliedReportingOverflow(by: inputChannels)
    guard !overflow else { throw WeightQuantizationError.sizeOverflow }
    guard values.count == count else {
      throw WeightQuantizationError.storageCountMismatch(
        name: "Q4 logical values", expected: count, actual: values.count)
    }
    for (index, value) in values.enumerated() where value < -8 || value > 7 {
      throw WeightQuantizationError.quantizedValueOutOfRange(
        index: index, value: Int(value), minimum: -8, maximum: 7)
    }
    return try GroupwiseQ4WeightMatrix(
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      groupSize: groupSize,
      packedValues: Array(repeating: 0, count: count / 2 + count % 2),
      scales: scales)
  }
}