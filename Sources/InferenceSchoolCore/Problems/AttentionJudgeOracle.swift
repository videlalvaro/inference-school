import Foundation

enum AttentionJudgeOracle {
  static func materialized(
    queries: FloatTensor,
    keys: FloatTensor,
    values: FloatTensor,
    configuration: AttentionConfiguration,
    window: Int? = nil
  ) throws -> FloatTensor {
    let input = try AttentionInput(
      queries: queries,
      keys: keys,
      values: values,
      configuration: configuration
    )
    if let window, window <= 0 { throw AttentionError.invalidWindow(window) }
    var output = Array(repeating: Float.zero, count: queries.elementCount)
    let scale = 1.0 / sqrt(Double(configuration.headDimension))

    for query in 0..<input.queryLength {
      let queryPosition = configuration.queryPositionOffset + query
      for queryHead in 0..<configuration.queryHeadCount {
        let keyValueHead = configuration.keyValueHead(forQueryHead: queryHead)
        var visibleKeys: [Int] = []
        var scores: [Double] = []
        for key in 0..<input.keyValueLength {
          let keyPosition = configuration.keyPositionOffset + key
          guard keyPosition <= queryPosition else { continue }
          if let window, keyPosition < queryPosition - window + 1 { continue }
          var dot = 0.0
          for feature in 0..<configuration.headDimension {
            dot +=
              Double(
                queries.storage[
                  input.queryOffset(
                    sequence: query,
                    head: queryHead,
                    feature: feature
                  )])
              * Double(
                keys.storage[
                  input.keyValueOffset(
                    sequence: key,
                    head: keyValueHead,
                    feature: feature
                  )])
          }
          visibleKeys.append(key)
          scores.append(dot * scale)
        }
        guard let maximum = scores.max() else {
          throw AttentionError.noVisibleKeys(queryPosition: queryPosition)
        }
        let exponentials = scores.map { exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        for feature in 0..<configuration.headDimension {
          var sum = 0.0
          for (index, key) in visibleKeys.enumerated() {
            sum +=
              exponentials[index] / denominator
              * Double(
                values.storage[
                  input.keyValueOffset(
                    sequence: key,
                    head: keyValueHead,
                    feature: feature
                  )
                ])
          }
          output[
            input.queryOffset(
              sequence: query,
              head: queryHead,
              feature: feature
            )] = Float(sum)
        }
      }
    }
    return try FloatTensor(output, shape: queries.shape)
  }

  static func approximatelyEqual(
    _ lhs: FloatTensor,
    _ rhs: FloatTensor,
    absoluteTolerance: Float = 3e-5,
    relativeTolerance: Float = 8e-5
  ) -> Bool {
    lhs.shape == rhs.shape
      && zip(lhs.storage, rhs.storage).allSatisfy { actual, expected in
        abs(actual - expected) <= absoluteTolerance + relativeTolerance * abs(expected)
      }
  }

  static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(
        JudgeFailure(
          caseName: name,
          message: "expected an error, but the implementation returned"
        ))
      return 0
    } catch {
      return 1
    }
  }
}
