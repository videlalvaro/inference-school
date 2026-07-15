import Foundation

public enum CheckpointParityError: Error, Equatable, LocalizedError {
  case unsupportedArtifactVersion(Int)
  case invalidArtifactJSON(String)
  case fingerprintMismatch(expected: String, actual: String)
  case promptMismatch
  case duplicateCapture(String)
  case emptyCaptureName(index: Int)
  case invalidCaptureShape(name: String)
  case captureElementCountMismatch(name: String, expected: Int, actual: Int)
  case nonFiniteCaptureValue(name: String, index: Int)
  case invalidTolerance(absolute: Float, relative: Float)
  case missingCapture(String)
  case unexpectedCapture(String)

  public var errorDescription: String? {
    switch self {
    case .unsupportedArtifactVersion(let value):
      "Reference artifact version \(value) is unsupported."
    case .invalidArtifactJSON(let message):
      "Reference artifact JSON is invalid: \(message)"
    case .fingerprintMismatch(let expected, let actual):
      "Reference artifact fingerprint \(actual) does not match model fingerprint \(expected)."
    case .promptMismatch:
      "Reference artifact token IDs or position offset do not match the parity request."
    case .duplicateCapture(let name):
      "Reference artifact capture '\(name)' appears more than once."
    case .emptyCaptureName(let index):
      "Reference artifact capture \(index) has an empty name."
    case .invalidCaptureShape(let name):
      "Reference artifact capture '\(name)' has a negative dimension or an overflowing shape."
    case .captureElementCountMismatch(let name, let expected, let actual):
      "Reference artifact capture '\(name)' requires \(expected) values; received \(actual)."
    case .nonFiniteCaptureValue(let name, let index):
      "Reference artifact capture '\(name)' has a non-finite value at index \(index)."
    case .invalidTolerance(let absolute, let relative):
      "Parity tolerances must be finite and nonnegative; received absolute=\(absolute), relative=\(relative)."
    case .missingCapture(let name):
      "Candidate execution did not produce capture '\(name)'."
    case .unexpectedCapture(let name):
      "Candidate execution produced unexpected capture '\(name)'."
    }
  }
}

public enum MiniDecoderParityFault: String, Sendable, Equatable, Codable {
  case none
  case ropePositionOffset
  case additiveRMSNormGamma
}

public struct MiniDecoderCapture: Sendable, Equatable {
  public let name: String
  public let tensor: FloatTensor

  public init(name: String, tensor: FloatTensor) {
    self.name = name
    self.tensor = tensor
  }
}

public struct MiniDecoderCaptureSet: Sendable, Equatable {
  public let modelFingerprint: String
  public let tokenIDs: [Int]
  public let positionOffset: Int
  public let captures: [MiniDecoderCapture]
  public let selectedTokenID: Int

  public init(
    modelFingerprint: String,
    tokenIDs: [Int],
    positionOffset: Int,
    captures: [MiniDecoderCapture],
    selectedTokenID: Int
  ) {
    self.modelFingerprint = modelFingerprint
    self.tokenIDs = tokenIDs
    self.positionOffset = positionOffset
    self.captures = captures
    self.selectedTokenID = selectedTokenID
  }

  public static func fromPrefill(
    _ result: PromptPrefillResult,
    model: MiniDecoderModel,
    tokenIDs: [Int],
    positionOffset: Int,
    selectedTokenID: Int
  ) throws -> MiniDecoderCaptureSet {
    let dimension = model.configuration.modelDimension
    var embeddingValues: [Float] = []
    for tokenID in tokenIDs {
      let start = tokenID * dimension
      embeddingValues.append(
        contentsOf: model.tokenEmbedding.storage[start..<(start + dimension)])
    }
    var captures = [MiniDecoderCapture(
      name: "embeddings",
      tensor: try FloatTensor(embeddingValues, shape: [tokenIDs.count, dimension]))]
    for layer in result.layers {
      let prefix = "layer.\(layer.layerIndex)"
      let block = layer.block.intermediates
      captures.append(contentsOf: [
        MiniDecoderCapture(name: "\(prefix).residual_input", tensor: layer.residualInput),
        MiniDecoderCapture(name: "\(prefix).attention_norm", tensor: block.attentionNormalized),
        MiniDecoderCapture(name: "\(prefix).query", tensor: block.queries),
        MiniDecoderCapture(name: "\(prefix).key", tensor: block.keys),
        MiniDecoderCapture(name: "\(prefix).value", tensor: block.values),
        MiniDecoderCapture(name: "\(prefix).rope.query", tensor: block.rotatedQueries),
        MiniDecoderCapture(name: "\(prefix).rope.key", tensor: block.rotatedKeys),
        MiniDecoderCapture(name: "\(prefix).attention", tensor: block.attentionHeads),
        MiniDecoderCapture(name: "\(prefix).attention_output", tensor: block.attentionProjection),
        MiniDecoderCapture(name: "\(prefix).post_attention", tensor: block.postAttentionResidual),
        MiniDecoderCapture(name: "\(prefix).mlp_norm", tensor: block.mlpNormalized),
        MiniDecoderCapture(name: "\(prefix).mlp.gate", tensor: block.gateProjection),
        MiniDecoderCapture(name: "\(prefix).mlp.up", tensor: block.upProjection),
        MiniDecoderCapture(name: "\(prefix).mlp.gated", tensor: block.gatedHidden),
        MiniDecoderCapture(name: "\(prefix).output", tensor: layer.block.state.residual),
      ])
    }
    captures.append(MiniDecoderCapture(name: "final_norm", tensor: result.finalNormalized))
    captures.append(MiniDecoderCapture(name: "logits", tensor: result.logits))
    captures.append(MiniDecoderCapture(
      name: "selected_token",
      tensor: try FloatTensor([Float(selectedTokenID)], shape: [1])))
    return MiniDecoderCaptureSet(
      modelFingerprint: model.fingerprint,
      tokenIDs: tokenIDs,
      positionOffset: positionOffset,
      captures: captures,
      selectedTokenID: selectedTokenID)
  }
}

public struct ReferenceCaptureTensor: Codable, Sendable, Equatable {
  public let name: String
  public let shape: [Int]
  public let values: [Float]

  public init(name: String, shape: [Int], values: [Float]) {
    self.name = name
    self.shape = shape
    self.values = values
  }
}

public struct MiniDecoderReferenceArtifact: Codable, Sendable, Equatable {
  public let formatVersion: Int
  public let provenance: String
  public let modelFingerprint: String
  public let tokenIDs: [Int]
  public let positionOffset: Int
  public let captures: [ReferenceCaptureTensor]
  public let selectedTokenID: Int

  public init(
    formatVersion: Int = 1,
    provenance: String,
    modelFingerprint: String,
    tokenIDs: [Int],
    positionOffset: Int,
    captures: [ReferenceCaptureTensor],
    selectedTokenID: Int
  ) {
    self.formatVersion = formatVersion
    self.provenance = provenance
    self.modelFingerprint = modelFingerprint
    self.tokenIDs = tokenIDs
    self.positionOffset = positionOffset
    self.captures = captures
    self.selectedTokenID = selectedTokenID
  }
}

public enum MiniDecoderReferenceArtifactCodec {
  public static func encode(_ artifact: MiniDecoderReferenceArtifact) throws -> [UInt8] {
    try validate(artifact)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return [UInt8](try encoder.encode(artifact))
  }

  public static func decode(
    _ bytes: [UInt8],
    expectedModelFingerprint: String? = nil
  ) throws -> MiniDecoderReferenceArtifact {
    let artifact: MiniDecoderReferenceArtifact
    do {
      artifact = try JSONDecoder().decode(
        MiniDecoderReferenceArtifact.self, from: Data(bytes))
    } catch {
      throw CheckpointParityError.invalidArtifactJSON(error.localizedDescription)
    }
    try validate(artifact)
    if let expectedModelFingerprint,
      artifact.modelFingerprint != expectedModelFingerprint
    {
      throw CheckpointParityError.fingerprintMismatch(
        expected: expectedModelFingerprint, actual: artifact.modelFingerprint)
    }
    return artifact
  }

  public static func validate(_ artifact: MiniDecoderReferenceArtifact) throws {
    guard artifact.formatVersion == 1 else {
      throw CheckpointParityError.unsupportedArtifactVersion(artifact.formatVersion)
    }
    var names: Set<String> = []
    for (index, capture) in artifact.captures.enumerated() {
      guard !capture.name.isEmpty else {
        throw CheckpointParityError.emptyCaptureName(index: index)
      }
      guard names.insert(capture.name).inserted else {
        throw CheckpointParityError.duplicateCapture(capture.name)
      }
      var count = 1
      for dimension in capture.shape {
        guard dimension >= 0 else {
          throw CheckpointParityError.invalidCaptureShape(name: capture.name)
        }
        let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
        guard !overflow else {
          throw CheckpointParityError.invalidCaptureShape(name: capture.name)
        }
        count = next
      }
      guard count == capture.values.count else {
        throw CheckpointParityError.captureElementCountMismatch(
          name: capture.name, expected: count, actual: capture.values.count)
      }
      if let valueIndex = capture.values.firstIndex(where: { !$0.isFinite }) {
        throw CheckpointParityError.nonFiniteCaptureValue(
          name: capture.name, index: valueIndex)
      }
    }
  }
}

public struct CaptureComparison: Sendable, Equatable {
  public let name: String
  public let referenceShape: [Int]
  public let candidateShape: [Int]?
  public let maximumAbsoluteError: Double?
  public let rootMeanSquareError: Double?
  public let cosineSimilarity: Double?
  public let argmaxMatches: Bool?
  public let passesTolerance: Bool

  public init(
    name: String,
    referenceShape: [Int],
    candidateShape: [Int]?,
    maximumAbsoluteError: Double?,
    rootMeanSquareError: Double?,
    cosineSimilarity: Double?,
    argmaxMatches: Bool?,
    passesTolerance: Bool
  ) {
    self.name = name
    self.referenceShape = referenceShape
    self.candidateShape = candidateShape
    self.maximumAbsoluteError = maximumAbsoluteError
    self.rootMeanSquareError = rootMeanSquareError
    self.cosineSimilarity = cosineSimilarity
    self.argmaxMatches = argmaxMatches
    self.passesTolerance = passesTolerance
  }
}

public struct CheckpointParityReport: Sendable, Equatable {
  public let modelFingerprint: String
  public let artifactProvenance: String
  public let comparisons: [CaptureComparison]
  public let firstDivergentCapture: String?
  public let referenceSelectedTokenID: Int
  public let candidateSelectedTokenID: Int
  public let selectedTokenMatches: Bool
  public let isPassing: Bool

  public init(
    modelFingerprint: String,
    artifactProvenance: String,
    comparisons: [CaptureComparison],
    firstDivergentCapture: String?,
    referenceSelectedTokenID: Int,
    candidateSelectedTokenID: Int,
    selectedTokenMatches: Bool,
    isPassing: Bool
  ) {
    self.modelFingerprint = modelFingerprint
    self.artifactProvenance = artifactProvenance
    self.comparisons = comparisons
    self.firstDivergentCapture = firstDivergentCapture
    self.referenceSelectedTokenID = referenceSelectedTokenID
    self.candidateSelectedTokenID = candidateSelectedTokenID
    self.selectedTokenMatches = selectedTokenMatches
    self.isPassing = isPassing
  }
}

public struct CheckpointParityRequest: Sendable, Equatable {
  public let model: MiniDecoderModel
  public let tokenIDs: [Int]
  public let positionOffset: Int
  public let referenceArtifactBytes: [UInt8]
  public let fault: MiniDecoderParityFault
  public let absoluteTolerance: Float
  public let relativeTolerance: Float

  public init(
    model: MiniDecoderModel,
    tokenIDs: [Int],
    positionOffset: Int,
    referenceArtifactBytes: [UInt8],
    fault: MiniDecoderParityFault = .none,
    absoluteTolerance: Float = 8e-5,
    relativeTolerance: Float = 1.5e-4
  ) {
    self.model = model
    self.tokenIDs = tokenIDs
    self.positionOffset = positionOffset
    self.referenceArtifactBytes = referenceArtifactBytes
    self.fault = fault
    self.absoluteTolerance = absoluteTolerance
    self.relativeTolerance = relativeTolerance
  }
}

public typealias CheckpointParityImplementation = (
  _ request: CheckpointParityRequest
) throws -> CheckpointParityReport

public enum P042CheckpointParityContract {
  public static func validate(
    _ request: CheckpointParityRequest
  ) throws -> MiniDecoderReferenceArtifact {
    try request.model.validate(tokenIDs: request.tokenIDs)
    guard request.positionOffset >= 0 else {
      throw DecoderBlockError.invalidPositionOffset(request.positionOffset)
    }
    guard request.absoluteTolerance.isFinite,
      request.relativeTolerance.isFinite,
      request.absoluteTolerance >= 0,
      request.relativeTolerance >= 0
    else {
      throw CheckpointParityError.invalidTolerance(
        absolute: request.absoluteTolerance, relative: request.relativeTolerance)
    }
    let artifact = try MiniDecoderReferenceArtifactCodec.decode(
      request.referenceArtifactBytes,
      expectedModelFingerprint: request.model.fingerprint)
    guard artifact.tokenIDs == request.tokenIDs,
      artifact.positionOffset == request.positionOffset
    else { throw CheckpointParityError.promptMismatch }
    return artifact
  }
}

public enum P042EducationalReference {
  public static let provenance =
    "educational-mini-model fixture generated by Inference School's independent Double CPU oracle"

  public static func artifact(
    model: MiniDecoderModel,
    tokenIDs: [Int],
    positionOffset: Int = 0
  ) throws -> MiniDecoderReferenceArtifact {
    let result = try MiniDecoderReference.prefill(PromptPrefillRequest(
      model: model, tokenIDs: tokenIDs, positionOffset: positionOffset))
    let selected = greedyToken(result.logits.storage)
    let captureSet = try MiniDecoderCaptureSet.fromPrefill(
      result,
      model: model,
      tokenIDs: tokenIDs,
      positionOffset: positionOffset,
      selectedTokenID: selected)
    return MiniDecoderReferenceArtifact(
      provenance: provenance,
      modelFingerprint: model.fingerprint,
      tokenIDs: tokenIDs,
      positionOffset: positionOffset,
      captures: captureSet.captures.map {
        ReferenceCaptureTensor(
          name: $0.name, shape: $0.tensor.shape, values: $0.tensor.storage)
      },
      selectedTokenID: selected)
  }

  public static func artifactBytes(
    model: MiniDecoderModel,
    tokenIDs: [Int],
    positionOffset: Int = 0
  ) throws -> [UInt8] {
    try MiniDecoderReferenceArtifactCodec.encode(artifact(
      model: model, tokenIDs: tokenIDs, positionOffset: positionOffset))
  }

  private static func greedyToken(_ logits: [Float]) -> Int {
    logits.indices.dropFirst().reduce(0) { best, candidate in
      logits[candidate] > logits[best] ? candidate : best
    }
  }
}

public enum P042CheckpointParityJudge {
  public static func evaluate(_ implementation: CheckpointParityImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      let model = try EducationalMiniModelFixture.make()
      let tokens = EducationalMiniModelFixture.defaultPrompt
      let positionOffset = 4
      let bytes = try P042EducationalReference.artifactBytes(
        model: model, tokenIDs: tokens, positionOffset: positionOffset)
      let canonical = try implementation(CheckpointParityRequest(
        model: model,
        tokenIDs: tokens,
        positionOffset: positionOffset,
        referenceArtifactBytes: bytes))
      if canonical.isPassing,
        canonical.firstDivergentCapture == nil,
        canonical.selectedTokenMatches,
        canonical.comparisons.allSatisfy(\.passesTolerance)
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "complete mini-model capture parity",
          message: "all named captures and the selected token must match the independent artifact"))
      }

      let ropeFault = try implementation(CheckpointParityRequest(
        model: model,
        tokenIDs: tokens,
        positionOffset: positionOffset,
        referenceArtifactBytes: bytes,
        fault: .ropePositionOffset))
      if !ropeFault.isPassing,
        ropeFault.firstDivergentCapture == "layer.0.rope.query"
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "RoPE fault localizes at first rotated query",
          message: "the first divergence must be layer.0.rope.query, not only final logits"))
      }

      let normFault = try implementation(CheckpointParityRequest(
        model: model,
        tokenIDs: tokens,
        positionOffset: positionOffset,
        referenceArtifactBytes: bytes,
        fault: .additiveRMSNormGamma))
      if !normFault.isPassing,
        normFault.firstDivergentCapture == "layer.0.attention_norm"
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "RMSNorm convention fault localizes at first norm",
          message: "the first divergence must be layer.0.attention_norm"))
      }

      let artifact = try MiniDecoderReferenceArtifactCodec.decode(
        bytes, expectedModelFingerprint: model.fingerprint)
      let roundTrip = try MiniDecoderReferenceArtifactCodec.encode(artifact)
      if bytes == roundTrip,
        artifact.provenance == P042EducationalReference.provenance,
        artifact.captures.count == canonical.comparisons.count
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "deterministic educational reference artifact",
          message: "sorted-key JSON round-trip, provenance, or capture count differs"))
      }

      passed += expectError(name: "reject stale model fingerprint", failures: &failures) {
        let otherModel = try EducationalMiniModelFixture.make(layerCount: 1)
        _ = try implementation(CheckpointParityRequest(
          model: otherModel,
          tokenIDs: tokens,
          positionOffset: positionOffset,
          referenceArtifactBytes: bytes))
      }
      passed += expectError(name: "reject artifact for another prompt", failures: &failures) {
        _ = try implementation(CheckpointParityRequest(
          model: model,
          tokenIDs: [1, 4],
          positionOffset: positionOffset,
          referenceArtifactBytes: bytes))
      }
      passed += expectError(name: "reject invalid tolerance", failures: &failures) {
        _ = try implementation(CheckpointParityRequest(
          model: model,
          tokenIDs: tokens,
          positionOffset: positionOffset,
          referenceArtifactBytes: bytes,
          absoluteTolerance: -.infinity))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 7, failures: failures)
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