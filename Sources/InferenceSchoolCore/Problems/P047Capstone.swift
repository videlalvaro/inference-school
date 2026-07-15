import Foundation

public enum CapstoneError: Error, Equatable, LocalizedError {
  case invalidTokenizerVocabulary(expected: [Int], actual: [Int])
  case tokenizerRoundTripMismatch
  case emptyPromptText
  case invalidGeneratedToken(tokenID: Int)
  case invalidReport(String)

  public var errorDescription: String? {
    switch self {
    case let .invalidTokenizerVocabulary(expected, actual):
      "Capstone tokenizer IDs must exactly match model IDs \(expected); received \(actual)."
    case .tokenizerRoundTripMismatch:
      "Capstone tokenizer encode/decode did not preserve the prompt bytes."
    case .emptyPromptText:
      "Capstone prompt text must not be empty."
    case let .invalidGeneratedToken(tokenID):
      "Generated token ID \(tokenID) is outside the capstone model vocabulary."
    case let .invalidReport(message):
      message
    }
  }
}

public enum GeneratedRendering: Sendable, Equatable {
  case text(String)
  case hexadecimal(String)
}

public struct CapstoneStageTiming: Sendable, Equatable {
  public let name: String
  public let nanoseconds: UInt64

  public init(name: String, nanoseconds: UInt64) {
    self.name = name
    self.nanoseconds = nanoseconds
  }
}

public struct CapstoneParityCapture: Sendable, Equatable {
  public let name: String
  public let maximumAbsoluteError: Double
  public let passes: Bool

  public init(name: String, maximumAbsoluteError: Double, passes: Bool) {
    self.name = name
    self.maximumAbsoluteError = maximumAbsoluteError
    self.passes = passes
  }
}

public struct MetalVerificationResources: Sendable, Equatable {
  public let allocatedBufferBytes: Int
  public let hostToDeviceBytes: Int
  public let deviceToHostBytes: Int
  public let dispatchCount: Int
  public let commandBufferCount: Int
  public let hostWaitCount: Int

  public init(
    allocatedBufferBytes: Int,
    hostToDeviceBytes: Int,
    deviceToHostBytes: Int,
    dispatchCount: Int,
    commandBufferCount: Int,
    hostWaitCount: Int
  ) {
    self.allocatedBufferBytes = allocatedBufferBytes
    self.hostToDeviceBytes = hostToDeviceBytes
    self.deviceToHostBytes = deviceToHostBytes
    self.dispatchCount = dispatchCount
    self.commandBufferCount = commandBufferCount
    self.hostWaitCount = hostWaitCount
  }
}

public enum MetalVerificationStatus: Sendable, Equatable {
  case notRequested
  case unavailable(String)
  case completed
}

public struct CapstoneMetalVerification: Sendable, Equatable {
  public let label: String
  public let status: MetalVerificationStatus
  public let captures: [CapstoneParityCapture]
  public let resources: MetalVerificationResources?

  public init(
    label: String,
    status: MetalVerificationStatus,
    captures: [CapstoneParityCapture],
    resources: MetalVerificationResources?
  ) {
    self.label = label
    self.status = status
    self.captures = captures
    self.resources = resources
  }

  public var parityPassed: Bool {
    status == .completed && !captures.isEmpty && captures.allSatisfy(\.passes)
  }
}

public struct CapstoneOptimizationComparison: Sendable, Equatable {
  public let name: String
  public let baselineDispatchCount: Int
  public let optimizedDispatchCount: Int
  public let baselineLogicalBytes: Int
  public let optimizedLogicalBytes: Int
  public let basis: String

  public init(
    name: String,
    baselineDispatchCount: Int,
    optimizedDispatchCount: Int,
    baselineLogicalBytes: Int,
    optimizedLogicalBytes: Int,
    basis: String
  ) {
    self.name = name
    self.baselineDispatchCount = baselineDispatchCount
    self.optimizedDispatchCount = optimizedDispatchCount
    self.baselineLogicalBytes = baselineLogicalBytes
    self.optimizedLogicalBytes = optimizedLogicalBytes
    self.basis = basis
  }
}

public struct CapstoneRejectedOptimization: Sendable, Equatable {
  public let name: String
  public let evidence: String

  public init(name: String, evidence: String) {
    self.name = name
    self.evidence = evidence
  }
}

public struct CapstoneRequest: Sendable, Equatable {
  public let model: MiniDecoderModel
  public let tokenizer: ByteBPETokenizer
  public let prompt: String
  public let maxNewTokens: Int
  public let seed: UInt64
  public let samplingStrategy: SamplingStrategy
  public let includeMetalVerification: Bool

  public init(
    model: MiniDecoderModel,
    tokenizer: ByteBPETokenizer,
    prompt: String,
    maxNewTokens: Int,
    seed: UInt64,
    samplingStrategy: SamplingStrategy,
    includeMetalVerification: Bool
  ) {
    self.model = model
    self.tokenizer = tokenizer
    self.prompt = prompt
    self.maxNewTokens = maxNewTokens
    self.seed = seed
    self.samplingStrategy = samplingStrategy
    self.includeMetalVerification = includeMetalVerification
  }
}

public struct CapstoneReport: Sendable, Equatable {
  public let prompt: String
  public let promptTokenIDs: [Int]
  public let generatedTokenIDs: [Int]
  public let generatedBytes: [UInt8]
  public let rendering: GeneratedRendering
  public let stopReason: GenerationStopReason
  public let timings: [CapstoneStageTiming]
  public let timeToFirstTokenNanoseconds: UInt64?
  public let decodeTokensPerSecond: Double?
  public let finalCacheCounts: [Int]
  public let modelWeightBytes: Int
  public let allocatedKVCacheBytes: Int
  public let prefillArenaBytes: Int
  public let decodeArenaBytes: Int
  public let generationBackend: String
  public let weightFormat: String
  public let keyValueFormat: String
  public let metalVerification: CapstoneMetalVerification
  public let optimizationComparison: CapstoneOptimizationComparison
  public let rejectedOptimization: CapstoneRejectedOptimization
  public let limitations: [String]

  public init(
    prompt: String,
    promptTokenIDs: [Int],
    generatedTokenIDs: [Int],
    generatedBytes: [UInt8],
    rendering: GeneratedRendering,
    stopReason: GenerationStopReason,
    timings: [CapstoneStageTiming],
    timeToFirstTokenNanoseconds: UInt64?,
    decodeTokensPerSecond: Double?,
    finalCacheCounts: [Int],
    modelWeightBytes: Int,
    allocatedKVCacheBytes: Int,
    prefillArenaBytes: Int,
    decodeArenaBytes: Int,
    generationBackend: String,
    weightFormat: String,
    keyValueFormat: String,
    metalVerification: CapstoneMetalVerification,
    optimizationComparison: CapstoneOptimizationComparison,
    rejectedOptimization: CapstoneRejectedOptimization,
    limitations: [String]
  ) {
    self.prompt = prompt
    self.promptTokenIDs = promptTokenIDs
    self.generatedTokenIDs = generatedTokenIDs
    self.generatedBytes = generatedBytes
    self.rendering = rendering
    self.stopReason = stopReason
    self.timings = timings
    self.timeToFirstTokenNanoseconds = timeToFirstTokenNanoseconds
    self.decodeTokensPerSecond = decodeTokensPerSecond
    self.finalCacheCounts = finalCacheCounts
    self.modelWeightBytes = modelWeightBytes
    self.allocatedKVCacheBytes = allocatedKVCacheBytes
    self.prefillArenaBytes = prefillArenaBytes
    self.decodeArenaBytes = decodeArenaBytes
    self.generationBackend = generationBackend
    self.weightFormat = weightFormat
    self.keyValueFormat = keyValueFormat
    self.metalVerification = metalVerification
    self.optimizationComparison = optimizationComparison
    self.rejectedOptimization = rejectedOptimization
    self.limitations = limitations
  }
}

public typealias CapstoneImplementation = (CapstoneRequest) throws -> CapstoneReport

public enum P047CapstoneFixture {
  public static let defaultPrompt = "ab c."

  public static func makeTokenizer() throws -> ByteBPETokenizer {
    try ByteBPETokenizer(
      vocabulary: [
        BPEVocabularyToken(id: 0, bytes: Array("<EOS>".utf8)),
        BPEVocabularyToken(id: 1, bytes: Array("<BOS>".utf8)),
        BPEVocabularyToken(id: 2, bytes: Array("a".utf8)),
        BPEVocabularyToken(id: 3, bytes: Array("b".utf8)),
        BPEVocabularyToken(id: 4, bytes: Array("c".utf8)),
        BPEVocabularyToken(id: 5, bytes: Array(" ".utf8)),
        BPEVocabularyToken(id: 6, bytes: Array(".".utf8)),
      ],
      merges: [],
      beginningOfSequenceTokenID: 1,
      endOfSequenceTokenID: 0,
      unknownBytePolicy: .error)
  }

  public static func makeRequest(
    prompt: String = defaultPrompt,
    maxNewTokens: Int = 4,
    seed: UInt64 = 47,
    includeMetalVerification: Bool = false
  ) throws -> CapstoneRequest {
    CapstoneRequest(
      model: try EducationalMiniModelFixture.make(),
      tokenizer: try makeTokenizer(),
      prompt: prompt,
      maxNewTokens: maxNewTokens,
      seed: seed,
      samplingStrategy: .stochastic(SamplingConfiguration(
        temperature: 0.8, topK: 5, topP: 0.9)),
      includeMetalVerification: includeMetalVerification)
  }
}

public enum P047CapstoneContract {
  public static let generationBackend = "CPU reference backend"
  public static let metalVerificationLabel = "Metal verification slice (fused QKV + RoPE)"

  public static func validate(_ request: CapstoneRequest) throws {
    guard !request.prompt.isEmpty else { throw CapstoneError.emptyPromptText }
    guard request.maxNewTokens >= 0 else {
      throw MiniDecoderError.invalidGenerationLimit(request.maxNewTokens)
    }
    let expected = Array(0..<request.model.vocabularySize)
    let actual = request.tokenizer.tokensByID.keys.sorted()
    guard actual == expected else {
      throw CapstoneError.invalidTokenizerVocabulary(expected: expected, actual: actual)
    }
    guard request.tokenizer.endOfSequenceTokenID < request.model.vocabularySize,
      request.tokenizer.beginningOfSequenceTokenID < request.model.vocabularySize
    else {
      throw CapstoneError.invalidTokenizerVocabulary(expected: expected, actual: actual)
    }
    try P038LogitsSamplingContract.validate(
      logits: Array(repeating: 0, count: request.model.vocabularySize),
      strategy: request.samplingStrategy)
  }

  public static func validate(_ report: CapstoneReport, for request: CapstoneRequest) throws {
    guard report.prompt == request.prompt else {
      throw CapstoneError.invalidReport("Report prompt does not match the request.")
    }
    try request.model.validate(tokenIDs: report.promptTokenIDs)
    for tokenID in report.generatedTokenIDs
      where tokenID < 0 || tokenID >= request.model.vocabularySize
    {
      throw CapstoneError.invalidGeneratedToken(tokenID: tokenID)
    }
    guard report.generatedTokenIDs.count <= request.maxNewTokens else {
      throw CapstoneError.invalidReport("Generated more tokens than maxNewTokens.")
    }
    let decodedPrompt = request.tokenizer.tokensByID.isEmpty
      ? []
      : report.promptTokenIDs.flatMap { tokenID -> [UInt8] in
        if tokenID == request.tokenizer.beginningOfSequenceTokenID
          || tokenID == request.tokenizer.endOfSequenceTokenID
        { return [] }
        return request.tokenizer.tokensByID[tokenID] ?? []
      }
    guard decodedPrompt == Array(request.prompt.utf8) else {
      throw CapstoneError.tokenizerRoundTripMismatch
    }
    let expectedDecodeSteps = max(0, report.generatedTokenIDs.count - 1)
    let decodeTimings = report.timings.filter { $0.name.hasPrefix("decode.token.") }
    guard report.timings.first?.name == "prefill.engine",
      report.timings.allSatisfy({ $0.nanoseconds > 0 }),
      decodeTimings.count == expectedDecodeSteps
    else {
      throw CapstoneError.invalidReport("Stage timing names or sample counts are invalid.")
    }
    let expectedCacheCount = report.promptTokenIDs.count + expectedDecodeSteps
    guard report.finalCacheCounts.count == request.model.layerCount,
      report.finalCacheCounts.allSatisfy({ $0 == expectedCacheCount })
    else {
      throw CapstoneError.invalidReport("Final KV cache counts do not match serial decode growth.")
    }
    guard report.modelWeightBytes > 0,
      report.allocatedKVCacheBytes > 0,
      report.prefillArenaBytes > 0,
      report.decodeArenaBytes > 0,
      report.generationBackend == generationBackend,
      report.weightFormat == "Float32 row-major",
      report.keyValueFormat == "Float32 contiguous KV cache",
      report.optimizationComparison.baselineDispatchCount
        > report.optimizationComparison.optimizedDispatchCount,
      report.optimizationComparison.baselineLogicalBytes
        > report.optimizationComparison.optimizedLogicalBytes,
      !report.rejectedOptimization.evidence.isEmpty,
      report.limitations.count >= 4
    else {
      throw CapstoneError.invalidReport("Memory, backend, optimization, or limitation fields are incomplete.")
    }
    if request.includeMetalVerification {
      switch report.metalVerification.status {
      case .completed:
        guard report.metalVerification.parityPassed,
          report.metalVerification.captures.map(\.name) == [
            "layer.0.fused_qkv.query",
            "layer.0.fused_qkv.key",
            "layer.0.fused_qkv.value",
            "layer.0.rope.query",
            "layer.0.rope.key",
          ],
          report.metalVerification.resources?.dispatchCount == 3
        else {
          throw CapstoneError.invalidReport("Metal verification captures or resource counts failed.")
        }
      case .unavailable:
        break
      case .notRequested:
        throw CapstoneError.invalidReport("Metal verification was requested but not attempted.")
      }
    } else if report.metalVerification.status != .notRequested {
      throw CapstoneError.invalidReport("Metal verification ran when it was not requested.")
    }
  }
}

public enum P047CapstoneJudge {
  public static func evaluate(_ implementation: CapstoneImplementation) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      let request = try P047CapstoneFixture.makeRequest(
        maxNewTokens: 4, seed: 47, includeMetalVerification: false)
      let first = try implementation(request)
      try P047CapstoneContract.validate(first, for: request)
      let second = try implementation(request)
      try P047CapstoneContract.validate(second, for: request)
      if first.promptTokenIDs == [1, 2, 3, 5, 4, 6],
        first.generatedTokenIDs == second.generatedTokenIDs,
        first.stopReason == second.stopReason,
        first.generatedBytes == second.generatedBytes
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "deterministic end-to-end generation",
          message: "prompt tokenization, generated IDs, stop reason, or bytes changed for the same seed"))
      }
      let oracle = try referenceGeneration(
        request: request,
        promptTokenIDs: first.promptTokenIDs)
      if first.generatedTokenIDs == oracle.tokenIDs,
        first.stopReason == oracle.stopReason
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "cached decode matches full-prefix oracle",
          message: "serial KV-cache generation differs from independent full-prefix recomputation"))
      }
      if first.finalCacheCounts.allSatisfy({
        $0 == first.promptTokenIDs.count + max(0, first.generatedTokenIDs.count - 1)
      }) {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "cache grows once per serial decode step",
          message: "cache count must equal prompt tokens plus decode engine calls"))
      }
      if first.generationBackend == P047CapstoneContract.generationBackend,
        first.metalVerification.status == .notRequested,
        first.limitations.contains(where: { $0.contains("not pretrained") }),
        first.limitations.contains(where: { $0.contains("restricted") })
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "honest backend and model scope",
          message: "report must identify the CPU backend, verification-only Metal, and fixture limits"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "capstone execution", message: error.localizedDescription))
    }

    passed += expectError(name: "reject unsupported prompt byte", failures: &failures) {
      _ = try implementation(P047CapstoneFixture.makeRequest(prompt: "d"))
    }
    passed += expectError(name: "reject incompatible tokenizer vocabulary", failures: &failures) {
      let model = try EducationalMiniModelFixture.make()
      let tokenizer = try P037ByteBPEFixture.makeTokenizer()
      _ = try implementation(CapstoneRequest(
        model: model,
        tokenizer: tokenizer,
        prompt: "a",
        maxNewTokens: 1,
        seed: 1,
        samplingStrategy: .greedy,
        includeMetalVerification: false))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 6, failures: failures)
  }

  private static func referenceGeneration(
    request: CapstoneRequest,
    promptTokenIDs: [Int]
  ) throws -> (tokenIDs: [Int], stopReason: GenerationStopReason) {
    var prefix = promptTokenIDs
    var generated: [Int] = []
    var generator = SeededGenerator(seed: request.seed)
    while generated.count < request.maxNewTokens {
      let result = try MiniDecoderReference.prefill(PromptPrefillRequest(
        model: request.model,
        tokenIDs: prefix,
        positionOffset: 0))
      let tokenID = try referenceSample(
        logits: result.logits.storage,
        strategy: request.samplingStrategy,
        generator: &generator)
      generated.append(tokenID)
      if tokenID == request.tokenizer.endOfSequenceTokenID {
        return (generated, .endOfSequence)
      }
      prefix.append(tokenID)
    }
    return (generated, .maximumTokenCount)
  }

  private static func referenceSample(
    logits: [Float],
    strategy: SamplingStrategy,
    generator: inout SeededGenerator
  ) throws -> Int {
    try P038LogitsSamplingContract.validate(logits: logits, strategy: strategy)
    var ranked = logits.indices.map { (id: $0, logit: logits[$0]) }
    ranked.sort { lhs, rhs in
      lhs.logit == rhs.logit ? lhs.id < rhs.id : lhs.logit > rhs.logit
    }
    guard case .stochastic(let configuration) = strategy else { return ranked[0].id }
    if let topK = configuration.topK { ranked = Array(ranked.prefix(topK)) }
    let scaled = ranked.map { Double($0.logit) / Double(configuration.temperature) }
    let maximum = scaled.max()!
    var probabilities = scaled.map { exp($0 - maximum) }
    let total = probabilities.reduce(0, +)
    probabilities = probabilities.map { $0 / total }
    if let topP = configuration.topP {
      var cumulative = 0.0
      var retained = 0
      repeat {
        cumulative += probabilities[retained]
        retained += 1
      } while retained < probabilities.count && cumulative < Double(topP)
      ranked = Array(ranked.prefix(retained))
      probabilities = Array(probabilities.prefix(retained))
    }
    let retainedTotal = probabilities.reduce(0, +)
    probabilities = probabilities.map { $0 / retainedTotal }
    let draw = generator.nextUnitInterval()
    var cumulative = 0.0
    for index in ranked.indices {
      cumulative += probabilities[index]
      if draw < cumulative { return ranked[index].id }
    }
    return ranked.last!.id
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
