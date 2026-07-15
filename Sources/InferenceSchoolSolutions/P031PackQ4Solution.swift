import InferenceSchoolCore

public enum P031PackQ4Solution {
  public static func pack(
    _ values: [Int8],
    outputChannels: Int,
    inputChannels: Int,
    groupSize: Int,
    scales: [Float]
  ) throws -> GroupwiseQ4WeightMatrix {
    try WeightQuantizationSolutionSupport.packQ4(
      values,
      outputChannels: outputChannels,
      inputChannels: inputChannels,
      groupSize: groupSize,
      scales: scales)
  }
}