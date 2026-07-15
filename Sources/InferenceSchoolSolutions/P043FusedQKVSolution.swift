import Foundation
import InferenceSchoolCore

public enum P043FusedQKVSolution {
  public static func separate(_ request: FusedQKVRequest) throws -> FusedQKVResult {
    try P043FusedQKVContract.validate(request)
    let sequence = request.input.shape[0]
    let model = request.configuration.modelDimension
    var normalized = Array(repeating: Float.zero, count: request.input.elementCount)
    for token in 0..<sequence {
      var sumSquares: Float = 0
      for feature in 0..<model {
        let value = request.input.storage[token * model + feature]
        sumSquares += value * value
      }
      let inverseRMS = 1 / sqrt(sumSquares / Float(model) + request.epsilon)
      for feature in 0..<model {
        normalized[token * model + feature] =
          request.input.storage[token * model + feature] * inverseRMS
          * request.gamma.storage[feature]
      }
    }

    func project(_ weights: FloatTensor, heads: Int) throws -> FloatTensor {
      let outputWidth = weights.shape[0]
      var output = Array(repeating: Float.zero, count: sequence * outputWidth)
      for token in 0..<sequence {
        for channel in 0..<outputWidth {
          var sum: Float = 0
          for feature in 0..<model {
            sum += normalized[token * model + feature]
              * weights.storage[channel * model + feature]
          }
          output[token * outputWidth + channel] = sum
        }
      }
      return try FloatTensor(
        output, shape: [sequence, heads, request.configuration.headDimension])
    }

    return FusedQKVResult(
      queries: try project(
        request.queryWeights, heads: request.configuration.queryHeadCount),
      keys: try project(
        request.keyWeights, heads: request.configuration.keyValueHeadCount),
      values: try project(
        request.valueWeights, heads: request.configuration.keyValueHeadCount))
  }

  public static func fused(_ request: FusedQKVRequest) throws -> FusedQKVResult {
    try P043FusedQKVContract.validate(request)
    let sequence = request.input.shape[0]
    let model = request.configuration.modelDimension

    func project(_ weights: FloatTensor, heads: Int) throws -> FloatTensor {
      let outputWidth = weights.shape[0]
      var output = Array(repeating: Float.zero, count: sequence * outputWidth)
      for token in 0..<sequence {
        var sumSquares: Float = 0
        for feature in 0..<model {
          let value = request.input.storage[token * model + feature]
          sumSquares += value * value
        }
        let inverseRMS = 1 / sqrt(sumSquares / Float(model) + request.epsilon)
        for channel in 0..<outputWidth {
          var sum: Float = 0
          for feature in 0..<model {
            let normalized = request.input.storage[token * model + feature]
              * inverseRMS * request.gamma.storage[feature]
            sum += weights.storage[channel * model + feature] * normalized
          }
          output[token * outputWidth + channel] = sum
        }
      }
      return try FloatTensor(
        output, shape: [sequence, heads, request.configuration.headDimension])
    }

    return FusedQKVResult(
      queries: try project(
        request.queryWeights, heads: request.configuration.queryHeadCount),
      keys: try project(
        request.keyWeights, heads: request.configuration.keyValueHeadCount),
      values: try project(
        request.valueWeights, heads: request.configuration.keyValueHeadCount))
  }
}