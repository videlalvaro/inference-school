import Foundation
import InferenceSchoolCore

public enum P015RoPESolution {
  public static func apply(
    queries: FloatTensor,
    keys: FloatTensor,
    rotaryDimension: Int,
    base: Float,
    queryPositionOffset: Int,
    keyPositionOffset: Int
  ) throws -> RoPEResult {
    try RoPEContract.validate(
      queries: queries, keys: keys, rotaryDimension: rotaryDimension, base: base,
      queryPositionOffset: queryPositionOffset, keyPositionOffset: keyPositionOffset)
    func rotate(_ tensor: FloatTensor, offset: Int) throws -> FloatTensor {
      var output = tensor.storage
      let heads = tensor.shape[1]
      let headDimension = tensor.shape[2]
      for sequence in 0..<tensor.shape[0] {
        let position = Float(offset + sequence)
        for head in 0..<heads {
          let start = (sequence * heads + head) * headDimension
          for pairStart in stride(from: 0, to: rotaryDimension, by: 2) {
            let pair = pairStart / 2
            let angle = position / pow(base, Float(2 * pair) / Float(rotaryDimension))
            let cosine = cos(angle)
            let sine = sin(angle)
            let first = tensor.storage[start + pairStart]
            let second = tensor.storage[start + pairStart + 1]
            output[start + pairStart] = first * cosine - second * sine
            output[start + pairStart + 1] = first * sine + second * cosine
          }
        }
      }
      return try FloatTensor(output, shape: tensor.shape)
    }
    return RoPEResult(
      queries: try rotate(queries, offset: queryPositionOffset),
      keys: try rotate(keys, offset: keyPositionOffset)
    )
  }
}
