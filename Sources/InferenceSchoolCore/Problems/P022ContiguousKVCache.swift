import Foundation

public enum P022ContiguousKVCacheContract {
  public static func validate(
    _ append: KVCacheAppend,
    configuration: KVCacheConfiguration,
    counts: [Int],
    lastPositions: [Int?]
  ) throws {
    try configuration.validate(layer: append.layer)
    guard append.logicalPosition >= 0 else {
      throw KVCacheError.invalidLogicalPosition(append.logicalPosition)
    }
    try configuration.validate(vector: append.key, name: "Key")
    try configuration.validate(vector: append.value, name: "Value")
    guard counts[append.layer] < configuration.capacity else {
      throw KVCacheError.capacityExceeded(
        layer: append.layer, capacity: configuration.capacity)
    }
    if let lastPosition = lastPositions[append.layer] {
      let expected = lastPosition + 1
      guard append.logicalPosition == expected else {
        throw KVCacheError.positionSequenceMismatch(
          layer: append.layer, expected: expected, actual: append.logicalPosition)
      }
    }
  }
}

public enum P022ContiguousKVCacheJudge {
  public static func evaluate(_ implementation: ContiguousKVCacheImplementation) -> JudgeReport {
    var failures: [JudgeFailure] = []
    var passed = 0

    do {
      let configuration = try KVCacheConfiguration(
        layerCount: 2, keyValueHeadCount: 2, headDimension: 2, capacity: 3)
      let appends = [
        try record(layer: 0, position: 7, keyBase: 10, valueBase: 110),
        try record(layer: 1, position: 40, keyBase: 20, valueBase: 120),
        try record(layer: 0, position: 8, keyBase: 30, valueBase: 130),
      ]
      let actual = try implementation(configuration, appends)
      let expectedLayers = [
        try snapshot(
          positions: [7, 8], keyBases: [10, 30], valueBases: [110, 130]),
        try snapshot(positions: [40], keyBases: [20], valueBases: [120]),
      ]

      if actual.layers == expectedLayers {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "append, read, and layer isolation",
          message: "cache transcript differs from independent token-major fixture"))
      }
      if actual.allocatedBytes == 192,
        actual.keyStorageCount == 24,
        actual.valueStorageCount == 24,
        actual.storageAddressesStable
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "fixed allocation remains stable",
          message: "expected two fixed 24-float stores (192 bytes) with stable addresses"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "valid append sequence", message: error.localizedDescription))
    }

    passed += expectError(name: "reject capacity overflow", failures: &failures) {
      let configuration = try KVCacheConfiguration(
        layerCount: 1, keyValueHeadCount: 1, headDimension: 2, capacity: 1)
      _ = try implementation(configuration, [
        try record(layer: 0, position: 2, keyBase: 1, valueBase: 11, headCount: 1),
        try record(layer: 0, position: 3, keyBase: 2, valueBase: 12, headCount: 1),
      ])
    }
    passed += expectError(name: "reject logical position gap", failures: &failures) {
      let configuration = try KVCacheConfiguration(
        layerCount: 1, keyValueHeadCount: 1, headDimension: 2, capacity: 2)
      _ = try implementation(configuration, [
        try record(layer: 0, position: 5, keyBase: 1, valueBase: 11, headCount: 1),
        try record(layer: 0, position: 7, keyBase: 2, valueBase: 12, headCount: 1),
      ])
    }
    passed += expectError(name: "reject append shape", failures: &failures) {
      let configuration = try KVCacheConfiguration(
        layerCount: 1, keyValueHeadCount: 2, headDimension: 2, capacity: 2)
      let wrong = try FloatTensor([1, 2], shape: [1, 2])
      _ = try implementation(configuration, [
        KVCacheAppend(layer: 0, logicalPosition: 0, key: wrong, value: wrong)
      ])
    }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 5, failures: failures)
  }

  private static func record(
    layer: Int,
    position: Int,
    keyBase: Float,
    valueBase: Float,
    headCount: Int = 2
  ) throws -> KVCacheAppend {
    let count = headCount * 2
    return KVCacheAppend(
      layer: layer,
      logicalPosition: position,
      key: try FloatTensor((0..<count).map { keyBase + Float($0) }, shape: [headCount, 2]),
      value: try FloatTensor((0..<count).map { valueBase + Float($0) }, shape: [headCount, 2]))
  }

  private static func snapshot(
    positions: [Int], keyBases: [Float], valueBases: [Float]
  ) throws -> KVCacheLayerSnapshot {
    let keys = keyBases.flatMap { base in (0..<4).map { base + Float($0) } }
    let values = valueBases.flatMap { base in (0..<4).map { base + Float($0) } }
    return KVCacheLayerSnapshot(
      logicalPositions: positions,
      keys: try FloatTensor(keys, shape: [positions.count, 2, 2]),
      values: try FloatTensor(values, shape: [positions.count, 2, 2]))
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(caseName: name, message: "expected an error"))
      return 0
    } catch {
      return 1
    }
  }
}
