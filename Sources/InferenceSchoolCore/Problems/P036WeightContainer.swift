import Foundation

public enum WeightContainerError: Error, Equatable, LocalizedError {
  case truncatedPreamble(expected: Int, actual: Int)
  case invalidMagic
  case unsupportedVersion(UInt32)
  case headerLengthOverflow(UInt64)
  case headerOutOfBounds(end: Int, fileSize: Int)
  case invalidHeaderPadding(index: Int, value: UInt8)
  case invalidUTF8Header
  case invalidJSONHeader(String)
  case emptyTensorName(index: Int)
  case duplicateTensorName(String)
  case unsupportedDType(tensor: String, dtype: String)
  case invalidDimension(tensor: String, axis: Int, value: Int)
  case tensorSizeOverflow(String)
  case negativeOffset(tensor: String, value: Int)
  case misalignedOffset(tensor: String, offset: Int, alignment: Int)
  case negativeByteCount(tensor: String, value: Int)
  case byteCountMismatch(tensor: String, expected: Int, actual: Int)
  case payloadOutOfBounds(tensor: String, end: Int, payloadSize: Int)
  case overlappingTensors(first: String, second: String)
  case missingTensor(String)
  case nonFiniteTensorValue(tensor: String, linearIndex: Int)

  public var errorDescription: String? {
    switch self {
    case .truncatedPreamble(let expected, let actual):
      "InferenceWeight requires at least \(expected) preamble bytes; received \(actual)."
    case .invalidMagic:
      "InferenceWeight magic bytes do not match LLMWGT01."
    case .unsupportedVersion(let version):
      "InferenceWeight version \(version) is unsupported."
    case .headerLengthOverflow(let value):
      "Header length \(value) cannot be represented safely on this platform."
    case .headerOutOfBounds(let end, let fileSize):
      "Header ends at byte \(end), beyond file size \(fileSize)."
    case .invalidHeaderPadding(let index, let value):
      "Header alignment padding at byte \(index) must be zero; received \(value)."
    case .invalidUTF8Header:
      "InferenceWeight header is not valid UTF-8."
    case .invalidJSONHeader(let message):
      "InferenceWeight header is not valid schema-compatible JSON: \(message)"
    case .emptyTensorName(let index):
      "Tensor descriptor \(index) has an empty name."
    case .duplicateTensorName(let name):
      "Tensor name '\(name)' appears more than once."
    case .unsupportedDType(let tensor, let dtype):
      "Tensor '\(tensor)' uses unsupported dtype '\(dtype)'; expected f32-le."
    case .invalidDimension(let tensor, let axis, let value):
      "Tensor '\(tensor)' dimension \(axis) must be nonnegative; received \(value)."
    case .tensorSizeOverflow(let tensor):
      "Tensor '\(tensor)' dimensions or byte count exceed Int.max."
    case .negativeOffset(let tensor, let value):
      "Tensor '\(tensor)' offset must be nonnegative; received \(value)."
    case .misalignedOffset(let tensor, let offset, let alignment):
      "Tensor '\(tensor)' offset \(offset) must be aligned to \(alignment) bytes."
    case .negativeByteCount(let tensor, let value):
      "Tensor '\(tensor)' byte count must be nonnegative; received \(value)."
    case .byteCountMismatch(let tensor, let expected, let actual):
      "Tensor '\(tensor)' requires \(expected) bytes from dtype and shape; received \(actual)."
    case .payloadOutOfBounds(let tensor, let end, let payloadSize):
      "Tensor '\(tensor)' ends at payload byte \(end), beyond payload size \(payloadSize)."
    case .overlappingTensors(let first, let second):
      "Tensor payload ranges for '\(first)' and '\(second)' overlap."
    case .missingTensor(let name):
      "Required tensor '\(name)' is missing."
    case .nonFiniteTensorValue(let tensor, let linearIndex):
      "Tensor '\(tensor)' contains a non-finite Float32 at linear index \(linearIndex)."
    }
  }
}

public struct WeightModelMetadata: Codable, Sendable, Equatable {
  public let modelDimension: Int
  public let hiddenDimension: Int
  public let queryHeadCount: Int
  public let keyValueHeadCount: Int
  public let headDimension: Int
  public let rotaryDimension: Int
  public let rmsNormEpsilon: Float
  public let ropeBase: Float

  public init(configuration: DecoderConfiguration) {
    modelDimension = configuration.modelDimension
    hiddenDimension = configuration.hiddenDimension
    queryHeadCount = configuration.queryHeadCount
    keyValueHeadCount = configuration.keyValueHeadCount
    headDimension = configuration.headDimension
    rotaryDimension = configuration.rotaryDimension
    rmsNormEpsilon = configuration.rmsNormEpsilon
    ropeBase = configuration.ropeBase
  }

  public func decoderConfiguration() throws -> DecoderConfiguration {
    try DecoderConfiguration(
      modelDimension: modelDimension,
      hiddenDimension: hiddenDimension,
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: keyValueHeadCount,
      headDimension: headDimension,
      rotaryDimension: rotaryDimension,
      rmsNormEpsilon: rmsNormEpsilon,
      ropeBase: ropeBase)
  }
}

public struct WeightTensorMetadata: Codable, Sendable, Equatable {
  public let name: String
  public let dtype: String
  public let shape: [Int]
  public let offset: Int
  public let byteCount: Int

  public init(name: String, dtype: String, shape: [Int], offset: Int, byteCount: Int) {
    self.name = name
    self.dtype = dtype
    self.shape = shape
    self.offset = offset
    self.byteCount = byteCount
  }
}

public struct WeightContainerHeader: Codable, Sendable, Equatable {
  public let model: WeightModelMetadata
  public let tensors: [WeightTensorMetadata]

  public init(model: WeightModelMetadata, tensors: [WeightTensorMetadata]) {
    self.model = model
    self.tensors = tensors
  }
}

public struct WeightContainerPreamble: Sendable, Equatable {
  public let headerLength: Int
  public let headerEnd: Int
  public let payloadStart: Int

  public init(headerLength: Int, headerEnd: Int, payloadStart: Int) {
    self.headerLength = headerLength
    self.headerEnd = headerEnd
    self.payloadStart = payloadStart
  }
}

public struct ParsedWeightContainer: Sendable, Equatable {
  public let configuration: DecoderConfiguration
  public let tensorMetadata: [String: WeightTensorMetadata]
  public let tensors: [String: FloatTensor]
  public let payloadByteCount: Int

  public init(
    configuration: DecoderConfiguration,
    tensorMetadata: [String: WeightTensorMetadata],
    tensors: [String: FloatTensor],
    payloadByteCount: Int
  ) {
    self.configuration = configuration
    self.tensorMetadata = tensorMetadata
    self.tensors = tensors
    self.payloadByteCount = payloadByteCount
  }

  public func tensor(named name: String) throws -> FloatTensor {
    guard let tensor = tensors[name] else { throw WeightContainerError.missingTensor(name) }
    return tensor
  }
}

public struct LoadedDecoderBlock: Sendable, Equatable {
  public let configuration: DecoderConfiguration
  public let weights: DecoderBlockWeights

  public init(configuration: DecoderConfiguration, weights: DecoderBlockWeights) {
    self.configuration = configuration
    self.weights = weights
  }
}

public typealias WeightContainerImplementation = (
  _ bytes: [UInt8],
  _ requiredTensorNames: [String]
) throws -> ParsedWeightContainer

public enum P036WeightContainerContract {
  public static let magic = Array("LLMWGT01".utf8)
  public static let currentVersion: UInt32 = 1
  public static let preambleByteCount = 20
  public static let payloadAlignment = 8
  public static let float32Alignment = 4
  public static let float32DType = "f32-le"

  public static func validatePreamble(_ bytes: [UInt8]) throws -> WeightContainerPreamble {
    guard bytes.count >= preambleByteCount else {
      throw WeightContainerError.truncatedPreamble(
        expected: preambleByteCount, actual: bytes.count)
    }
    guard Array(bytes[0..<magic.count]) == magic else {
      throw WeightContainerError.invalidMagic
    }
    let version = readUInt32LittleEndian(bytes, at: 8)
    guard version == currentVersion else {
      throw WeightContainerError.unsupportedVersion(version)
    }
    let rawLength = readUInt64LittleEndian(bytes, at: 12)
    guard rawLength <= UInt64(Int.max) else {
      throw WeightContainerError.headerLengthOverflow(rawLength)
    }
    let headerLength = Int(rawLength)
    let (headerEnd, overflow) = preambleByteCount.addingReportingOverflow(headerLength)
    guard !overflow else { throw WeightContainerError.headerLengthOverflow(rawLength) }
    guard headerEnd <= bytes.count else {
      throw WeightContainerError.headerOutOfBounds(end: headerEnd, fileSize: bytes.count)
    }
    let payloadStart = try aligned(headerEnd, to: payloadAlignment)
    guard payloadStart <= bytes.count else {
      throw WeightContainerError.headerOutOfBounds(end: payloadStart, fileSize: bytes.count)
    }
    for index in headerEnd..<payloadStart where bytes[index] != 0 {
      throw WeightContainerError.invalidHeaderPadding(index: index, value: bytes[index])
    }
    return WeightContainerPreamble(
      headerLength: headerLength, headerEnd: headerEnd, payloadStart: payloadStart)
  }

  private static func aligned(_ value: Int, to alignment: Int) throws -> Int {
    let remainder = value % alignment
    guard remainder != 0 else { return value }
    let (result, overflow) = value.addingReportingOverflow(alignment - remainder)
    guard !overflow else { throw WeightContainerError.headerLengthOverflow(UInt64.max) }
    return result
  }

  private static func readUInt32LittleEndian(_ bytes: [UInt8], at offset: Int) -> UInt32 {
    UInt32(bytes[offset])
      | UInt32(bytes[offset + 1]) << 8
      | UInt32(bytes[offset + 2]) << 16
      | UInt32(bytes[offset + 3]) << 24
  }

  private static func readUInt64LittleEndian(_ bytes: [UInt8], at offset: Int) -> UInt64 {
    (0..<8).reduce(UInt64.zero) { partial, byte in
      partial | UInt64(bytes[offset + byte]) << UInt64(byte * 8)
    }
  }
}

public enum P036DecoderBlockLoader {
  public static let requiredTensorNames = [
    "block.attention_norm.weight",
    "block.attention.query.weight",
    "block.attention.key.weight",
    "block.attention.value.weight",
    "block.attention.output.weight",
    "block.mlp_norm.weight",
    "block.mlp.gate.weight",
    "block.mlp.up.weight",
    "block.mlp.down.weight",
  ]

  public static func load(_ container: ParsedWeightContainer) throws -> LoadedDecoderBlock {
    let weights = DecoderBlockWeights(
      attentionNormGamma: try container.tensor(named: requiredTensorNames[0]),
      queryWeights: try container.tensor(named: requiredTensorNames[1]),
      keyWeights: try container.tensor(named: requiredTensorNames[2]),
      valueWeights: try container.tensor(named: requiredTensorNames[3]),
      attentionOutputWeights: try container.tensor(named: requiredTensorNames[4]),
      mlpNormGamma: try container.tensor(named: requiredTensorNames[5]),
      gateWeights: try container.tensor(named: requiredTensorNames[6]),
      upWeights: try container.tensor(named: requiredTensorNames[7]),
      downWeights: try container.tensor(named: requiredTensorNames[8]))
    try P035DecoderBlockContract.validateWeights(
      weights, configuration: container.configuration)
    return LoadedDecoderBlock(configuration: container.configuration, weights: weights)
  }
}

public struct WeightFixtureTensor: Sendable, Equatable {
  public let name: String
  public let tensor: FloatTensor

  public init(name: String, tensor: FloatTensor) {
    self.name = name
    self.tensor = tensor
  }
}

public enum P036WeightFixtureEncoder {
  public static func encode(
    configuration: DecoderConfiguration,
    tensors: [WeightFixtureTensor]
  ) throws -> [UInt8] {
    var names: Set<String> = []
    var descriptors: [WeightTensorMetadata] = []
    var payload: [UInt8] = []
    for (index, fixture) in tensors.enumerated() {
      guard !fixture.name.isEmpty else {
        throw WeightContainerError.emptyTensorName(index: index)
      }
      guard names.insert(fixture.name).inserted else {
        throw WeightContainerError.duplicateTensorName(fixture.name)
      }
      while !payload.count.isMultiple(of: P036WeightContainerContract.float32Alignment) {
        payload.append(0)
      }
      let byteCount = try checkedMultiply(
        fixture.tensor.elementCount,
        MemoryLayout<Float>.size,
        tensor: fixture.name)
      descriptors.append(WeightTensorMetadata(
        name: fixture.name,
        dtype: P036WeightContainerContract.float32DType,
        shape: fixture.tensor.shape,
        offset: payload.count,
        byteCount: byteCount))
      for (linearIndex, value) in fixture.tensor.storage.enumerated() {
        guard value.isFinite else {
          throw WeightContainerError.nonFiniteTensorValue(
            tensor: fixture.name, linearIndex: linearIndex)
        }
        appendUInt32LittleEndian(value.bitPattern, to: &payload)
      }
    }
    let header = WeightContainerHeader(
      model: WeightModelMetadata(configuration: configuration), tensors: descriptors)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let headerBytes = [UInt8](try encoder.encode(header))
    var result = P036WeightContainerContract.magic
    appendUInt32LittleEndian(P036WeightContainerContract.currentVersion, to: &result)
    appendUInt64LittleEndian(UInt64(headerBytes.count), to: &result)
    result.append(contentsOf: headerBytes)
    while !result.count.isMultiple(of: P036WeightContainerContract.payloadAlignment) {
      result.append(0)
    }
    result.append(contentsOf: payload)
    return result
  }

  private static func checkedMultiply(_ lhs: Int, _ rhs: Int, tensor: String) throws -> Int {
    let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw WeightContainerError.tensorSizeOverflow(tensor) }
    return result
  }

  private static func appendUInt32LittleEndian(_ value: UInt32, to bytes: inout [UInt8]) {
    for shift in stride(from: 0, to: 32, by: 8) {
      bytes.append(UInt8(truncatingIfNeeded: value >> UInt32(shift)))
    }
  }

  private static func appendUInt64LittleEndian(_ value: UInt64, to bytes: inout [UInt8]) {
    for shift in stride(from: 0, to: 64, by: 8) {
      bytes.append(UInt8(truncatingIfNeeded: value >> UInt64(shift)))
    }
  }
}

public enum P036WeightContainerJudge {
  public static func evaluate(_ implementation: WeightContainerImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    let fixture: DecoderFixture
    do {
      fixture = try decoderFixture()
    } catch {
      return JudgeReport(
        passedCaseCount: 0,
        totalCaseCount: 13,
        failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)])
    }
    do {
      let parsed = try implementation(fixture.bytes, P036DecoderBlockLoader.requiredTensorNames)
      let loaded = try P036DecoderBlockLoader.load(parsed)
      if loaded.configuration == fixture.configuration,
        loaded.weights == fixture.weights,
        parsed.payloadByteCount > 0
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "valid decoder block archive",
          message: "parsed metadata, named Float32 tensors, or loaded block weights differ"))
      }
    } catch {
      failures.append(JudgeFailure(
        caseName: "valid decoder block archive", message: error.localizedDescription))
    }

    var invalidMagic = fixture.bytes
    invalidMagic[0] ^= 0xff
    passed += expectError(name: "reject invalid magic", failures: &failures) {
      _ = try implementation(invalidMagic, [])
    }
    var invalidVersion = fixture.bytes
    invalidVersion[8] = 2
    passed += expectError(name: "reject unsupported version", failures: &failures) {
      _ = try implementation(invalidVersion, [])
    }
    var invalidHeaderLength = fixture.bytes
    for index in 12..<20 { invalidHeaderLength[index] = 0xff }
    passed += expectError(name: "reject overflowing header length", failures: &failures) {
      _ = try implementation(invalidHeaderLength, [])
    }
    var invalidUTF8 = fixture.bytes
    invalidUTF8[P036WeightContainerContract.preambleByteCount] = 0xff
    passed += expectError(name: "reject invalid UTF-8", failures: &failures) {
      _ = try implementation(invalidUTF8, [])
    }

    let model = WeightModelMetadata(configuration: fixture.configuration)
    passed += expectError(name: "reject duplicate names", failures: &failures) {
      _ = try implementation(try rawContainer(
        model: model,
        descriptors: [
          WeightTensorMetadata(name: "x", dtype: "f32-le", shape: [1], offset: 0, byteCount: 4),
          WeightTensorMetadata(name: "x", dtype: "f32-le", shape: [1], offset: 4, byteCount: 4),
        ], payload: Array(repeating: 0, count: 8)), [])
    }
    passed += expectError(name: "reject unsupported dtype", failures: &failures) {
      _ = try implementation(try rawContainer(
        model: model,
        descriptors: [
          WeightTensorMetadata(name: "x", dtype: "f16-le", shape: [1], offset: 0, byteCount: 2)
        ], payload: [0, 0, 0, 0]), [])
    }
    passed += expectError(name: "reject byte-count mismatch", failures: &failures) {
      _ = try implementation(try rawContainer(
        model: model,
        descriptors: [
          WeightTensorMetadata(name: "x", dtype: "f32-le", shape: [2], offset: 0, byteCount: 4)
        ], payload: Array(repeating: 0, count: 8)), [])
    }
    passed += expectError(name: "reject misaligned offset", failures: &failures) {
      _ = try implementation(try rawContainer(
        model: model,
        descriptors: [
          WeightTensorMetadata(name: "x", dtype: "f32-le", shape: [1], offset: 2, byteCount: 4)
        ], payload: Array(repeating: 0, count: 8)), [])
    }
    passed += expectError(name: "reject overlapping ranges", failures: &failures) {
      _ = try implementation(try rawContainer(
        model: model,
        descriptors: [
          WeightTensorMetadata(name: "x", dtype: "f32-le", shape: [2], offset: 0, byteCount: 8),
          WeightTensorMetadata(name: "y", dtype: "f32-le", shape: [1], offset: 4, byteCount: 4),
        ], payload: Array(repeating: 0, count: 8)), [])
    }
    passed += expectError(name: "reject payload bounds", failures: &failures) {
      _ = try implementation(try rawContainer(
        model: model,
        descriptors: [
          WeightTensorMetadata(name: "x", dtype: "f32-le", shape: [1], offset: 4, byteCount: 4)
        ], payload: Array(repeating: 0, count: 4)), [])
    }
    passed += expectError(name: "reject missing required tensor", failures: &failures) {
      _ = try implementation(fixture.bytes, ["not.present"])
    }
    passed += expectError(name: "reject non-finite tensor", failures: &failures) {
      _ = try implementation(try rawContainer(
        model: model,
        descriptors: [
          WeightTensorMetadata(name: "x", dtype: "f32-le", shape: [1], offset: 0, byteCount: 4)
        ], payload: [0, 0, 0x80, 0x7f]), [])
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 13, failures: failures)
  }

  private struct DecoderFixture {
    let bytes: [UInt8]
    let configuration: DecoderConfiguration
    let weights: DecoderBlockWeights
  }

  private static func decoderFixture() throws -> DecoderFixture {
    let configuration = try DecoderConfiguration(
      modelDimension: 4,
      hiddenDimension: 6,
      queryHeadCount: 2,
      keyValueHeadCount: 1,
      headDimension: 2,
      rotaryDimension: 2,
      rmsNormEpsilon: 1e-5,
      ropeBase: 100)
    func tensor(_ shape: [Int], salt: Int) throws -> FloatTensor {
      let count = shape.reduce(1, *)
      return try FloatTensor((0..<count).map {
        Float((($0 * 5 + salt * 7) % 23) - 11) / 17
      }, shape: shape)
    }
    let weights = DecoderBlockWeights(
      attentionNormGamma: try tensor([4], salt: 1),
      queryWeights: try tensor([4, 4], salt: 2),
      keyWeights: try tensor([2, 4], salt: 3),
      valueWeights: try tensor([2, 4], salt: 4),
      attentionOutputWeights: try tensor([4, 4], salt: 5),
      mlpNormGamma: try tensor([4], salt: 6),
      gateWeights: try tensor([6, 4], salt: 7),
      upWeights: try tensor([6, 4], salt: 8),
      downWeights: try tensor([4, 6], salt: 9))
    let values = [
      weights.attentionNormGamma,
      weights.queryWeights,
      weights.keyWeights,
      weights.valueWeights,
      weights.attentionOutputWeights,
      weights.mlpNormGamma,
      weights.gateWeights,
      weights.upWeights,
      weights.downWeights,
    ]
    let fixtures = zip(P036DecoderBlockLoader.requiredTensorNames, values).map {
      WeightFixtureTensor(name: $0, tensor: $1)
    }
    return DecoderFixture(
      bytes: try P036WeightFixtureEncoder.encode(
        configuration: configuration, tensors: fixtures),
      configuration: configuration,
      weights: weights)
  }

  private static func rawContainer(
    model: WeightModelMetadata,
    descriptors: [WeightTensorMetadata],
    payload: [UInt8]
  ) throws -> [UInt8] {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let header = [UInt8](try encoder.encode(
      WeightContainerHeader(model: model, tensors: descriptors)))
    var bytes = P036WeightContainerContract.magic
    append(UInt32(P036WeightContainerContract.currentVersion), to: &bytes)
    append(UInt64(header.count), to: &bytes)
    bytes.append(contentsOf: header)
    while !bytes.count.isMultiple(of: P036WeightContainerContract.payloadAlignment) {
      bytes.append(0)
    }
    bytes.append(contentsOf: payload)
    return bytes
  }

  private static func append<T: FixedWidthInteger>(_ value: T, to bytes: inout [UInt8]) {
    for byte in 0..<MemoryLayout<T>.size {
      bytes.append(UInt8(truncatingIfNeeded: value >> T(byte * 8)))
    }
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(
        caseName: name, message: "expected an error, but the parser returned"))
      return 0
    } catch {
      return 1
    }
  }
}