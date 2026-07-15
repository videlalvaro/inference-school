import Foundation

public enum ProfilingError: Error, Equatable, LocalizedError {
  case emptySamples
  case invalidPercentile(Double)
  case invalidWarmupIterations(Int)
  case invalidMeasuredTrials(Int)
  case invalidDecodeSteps(Int)
  case invalidContextLength(Int)
  case nonPositiveLatency(stage: String)
  case invalidRate(stage: String, value: Double)
  case unexpectedSampleCount(stage: String, expected: Int, actual: Int)
  case backendMismatch(expected: String, actual: String)

  public var errorDescription: String? {
    switch self {
    case .emptySamples:
      "Latency statistics require at least one sample."
    case let .invalidPercentile(value):
      "Percentile must be in (0, 1]; received \(value)."
    case let .invalidWarmupIterations(value):
      "Warmup iteration count must be nonnegative; received \(value)."
    case let .invalidMeasuredTrials(value):
      "Measured trial count must be positive; received \(value)."
    case let .invalidDecodeSteps(value):
      "Decode steps per trial must be positive; received \(value)."
    case let .invalidContextLength(value):
      "Decode context lengths must be positive; received \(value)."
    case let .nonPositiveLatency(stage):
      "\(stage) contains a zero-duration latency sample."
    case let .invalidRate(stage, value):
      "\(stage) produced an invalid token rate \(value)."
    case let .unexpectedSampleCount(stage, expected, actual):
      "\(stage) expected \(expected) measured samples; received \(actual)."
    case let .backendMismatch(expected, actual):
      "Profiler backend must be \(expected); received \(actual)."
    }
  }
}

public struct LatencyStatistics: Sendable, Equatable {
  public let samplesNanoseconds: [UInt64]
  public let medianNanoseconds: Double
  public let percentile: Double
  public let percentileNanoseconds: UInt64
  public let minimumNanoseconds: UInt64
  public let maximumNanoseconds: UInt64

  public init(
    samplesNanoseconds: [UInt64],
    medianNanoseconds: Double,
    percentile: Double,
    percentileNanoseconds: UInt64,
    minimumNanoseconds: UInt64,
    maximumNanoseconds: UInt64
  ) {
    self.samplesNanoseconds = samplesNanoseconds
    self.medianNanoseconds = medianNanoseconds
    self.percentile = percentile
    self.percentileNanoseconds = percentileNanoseconds
    self.minimumNanoseconds = minimumNanoseconds
    self.maximumNanoseconds = maximumNanoseconds
  }
}

public enum P044LatencyStatistics {
  public static func summarize(
    _ samplesNanoseconds: [UInt64],
    percentile: Double = 0.95
  ) throws -> LatencyStatistics {
    guard !samplesNanoseconds.isEmpty else { throw ProfilingError.emptySamples }
    guard percentile > 0, percentile <= 1, percentile.isFinite else {
      throw ProfilingError.invalidPercentile(percentile)
    }
    let sorted = samplesNanoseconds.sorted()
    let middle = sorted.count / 2
    let median: Double
    if sorted.count.isMultiple(of: 2) {
      median = (Double(sorted[middle - 1]) + Double(sorted[middle])) / 2
    } else {
      median = Double(sorted[middle])
    }
    let nearestRank = max(1, Int(ceil(percentile * Double(sorted.count))))
    return LatencyStatistics(
      samplesNanoseconds: samplesNanoseconds,
      medianNanoseconds: median,
      percentile: percentile,
      percentileNanoseconds: sorted[nearestRank - 1],
      minimumNanoseconds: sorted[0],
      maximumNanoseconds: sorted[sorted.count - 1])
  }
}

public struct ProfilingWorkEstimate: Sendable, Equatable {
  public let floatingPointOperations: Int
  public let estimatedWeightBytesRead: Int
  public let cacheBytesWritten: Int

  public init(
    floatingPointOperations: Int,
    estimatedWeightBytesRead: Int,
    cacheBytesWritten: Int
  ) {
    self.floatingPointOperations = floatingPointOperations
    self.estimatedWeightBytesRead = estimatedWeightBytesRead
    self.cacheBytesWritten = cacheBytesWritten
  }

  public init(_ work: MiniDecoderWorkModel) {
    self.init(
      floatingPointOperations: work.projectionFLOPs + work.attentionFLOPs,
      estimatedWeightBytesRead: work.estimatedWeightBytesRead,
      cacheBytesWritten: work.cacheBytesWritten)
  }
}

public struct PrefillProfile: Sendable, Equatable {
  public let stageName: String
  public let promptTokenCount: Int
  public let latency: LatencyStatistics
  public let promptTokensPerSecond: Double
  public let work: ProfilingWorkEstimate

  public init(
    stageName: String,
    promptTokenCount: Int,
    latency: LatencyStatistics,
    promptTokensPerSecond: Double,
    work: ProfilingWorkEstimate
  ) {
    self.stageName = stageName
    self.promptTokenCount = promptTokenCount
    self.latency = latency
    self.promptTokensPerSecond = promptTokensPerSecond
    self.work = work
  }
}

public struct DecodeContextProfile: Sendable, Equatable {
  public let stageName: String
  public let initialContextLength: Int
  public let measuredTokenCount: Int
  public let perTokenLatency: LatencyStatistics
  public let decodeTokensPerSecond: Double
  public let averageWorkPerToken: ProfilingWorkEstimate

  public init(
    stageName: String,
    initialContextLength: Int,
    measuredTokenCount: Int,
    perTokenLatency: LatencyStatistics,
    decodeTokensPerSecond: Double,
    averageWorkPerToken: ProfilingWorkEstimate
  ) {
    self.stageName = stageName
    self.initialContextLength = initialContextLength
    self.measuredTokenCount = measuredTokenCount
    self.perTokenLatency = perTokenLatency
    self.decodeTokensPerSecond = decodeTokensPerSecond
    self.averageWorkPerToken = averageWorkPerToken
  }
}

public struct PrefillDecodeProfileReport: Sendable, Equatable {
  public let backend: String
  public let clock: String
  public let timingBoundary: String
  public let warmupIterations: Int
  public let measuredTrials: Int
  public let prefill: PrefillProfile
  public let decode: [DecodeContextProfile]

  public init(
    backend: String,
    clock: String,
    timingBoundary: String,
    warmupIterations: Int,
    measuredTrials: Int,
    prefill: PrefillProfile,
    decode: [DecodeContextProfile]
  ) {
    self.backend = backend
    self.clock = clock
    self.timingBoundary = timingBoundary
    self.warmupIterations = warmupIterations
    self.measuredTrials = measuredTrials
    self.prefill = prefill
    self.decode = decode
  }
}

public struct PrefillDecodeProfilingRequest: Sendable, Equatable {
  public let model: MiniDecoderModel
  public let promptTokenIDs: [Int]
  public let decodeContextLengths: [Int]
  public let warmupIterations: Int
  public let measuredTrials: Int
  public let decodeStepsPerTrial: Int
  public let percentile: Double
  public let seed: UInt64

  public init(
    model: MiniDecoderModel,
    promptTokenIDs: [Int],
    decodeContextLengths: [Int],
    warmupIterations: Int,
    measuredTrials: Int,
    decodeStepsPerTrial: Int,
    percentile: Double = 0.95,
    seed: UInt64 = 0x044
  ) {
    self.model = model
    self.promptTokenIDs = promptTokenIDs
    self.decodeContextLengths = decodeContextLengths
    self.warmupIterations = warmupIterations
    self.measuredTrials = measuredTrials
    self.decodeStepsPerTrial = decodeStepsPerTrial
    self.percentile = percentile
    self.seed = seed
  }
}

public typealias PrefillDecodeProfilingImplementation = (
  PrefillDecodeProfilingRequest
) throws -> PrefillDecodeProfileReport

public enum P044ProfilingContract {
  public static let cpuReferenceBackend = "CPU reference backend"

  public static func validate(_ request: PrefillDecodeProfilingRequest) throws {
    try request.model.validate(tokenIDs: request.promptTokenIDs)
    guard request.warmupIterations >= 0 else {
      throw ProfilingError.invalidWarmupIterations(request.warmupIterations)
    }
    guard request.measuredTrials > 0 else {
      throw ProfilingError.invalidMeasuredTrials(request.measuredTrials)
    }
    guard request.decodeStepsPerTrial > 0 else {
      throw ProfilingError.invalidDecodeSteps(request.decodeStepsPerTrial)
    }
    guard !request.decodeContextLengths.isEmpty else {
      throw ProfilingError.invalidContextLength(0)
    }
    for length in request.decodeContextLengths where length <= 0 {
      throw ProfilingError.invalidContextLength(length)
    }
    guard request.percentile > 0, request.percentile <= 1, request.percentile.isFinite else {
      throw ProfilingError.invalidPercentile(request.percentile)
    }
  }

  public static func validate(
    _ report: PrefillDecodeProfileReport,
    for request: PrefillDecodeProfilingRequest
  ) throws {
    guard report.backend == cpuReferenceBackend else {
      throw ProfilingError.backendMismatch(
        expected: cpuReferenceBackend, actual: report.backend)
    }
    try validate(
      report.prefill.latency,
      stage: report.prefill.stageName,
      expectedSamples: request.measuredTrials)
    guard report.prefill.promptTokenCount == request.promptTokenIDs.count else {
      throw ProfilingError.unexpectedSampleCount(
        stage: "prefill token count",
        expected: request.promptTokenIDs.count,
        actual: report.prefill.promptTokenCount)
    }
    try validateRate(report.prefill.promptTokensPerSecond, stage: report.prefill.stageName)
    guard report.decode.map(\.initialContextLength) == request.decodeContextLengths else {
      throw ProfilingError.invalidContextLength(report.decode.first?.initialContextLength ?? 0)
    }
    for profile in report.decode {
      let expected = request.measuredTrials * request.decodeStepsPerTrial
      try validate(profile.perTokenLatency, stage: profile.stageName, expectedSamples: expected)
      guard profile.measuredTokenCount == expected else {
        throw ProfilingError.unexpectedSampleCount(
          stage: profile.stageName, expected: expected, actual: profile.measuredTokenCount)
      }
      try validateRate(profile.decodeTokensPerSecond, stage: profile.stageName)
    }
  }

  private static func validate(
    _ statistics: LatencyStatistics,
    stage: String,
    expectedSamples: Int
  ) throws {
    guard statistics.samplesNanoseconds.count == expectedSamples else {
      throw ProfilingError.unexpectedSampleCount(
        stage: stage,
        expected: expectedSamples,
        actual: statistics.samplesNanoseconds.count)
    }
    guard statistics.samplesNanoseconds.allSatisfy({ $0 > 0 }) else {
      throw ProfilingError.nonPositiveLatency(stage: stage)
    }
  }

  private static func validateRate(_ rate: Double, stage: String) throws {
    guard rate.isFinite, rate > 0 else {
      throw ProfilingError.invalidRate(stage: stage, value: rate)
    }
  }
}

public enum P044ProfilingJudge {
  public static func evaluate(
    _ implementation: PrefillDecodeProfilingImplementation
  ) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []

    do {
      let odd = try P044LatencyStatistics.summarize([30, 10, 20], percentile: 0.95)
      if odd.medianNanoseconds == 20, odd.percentileNanoseconds == 30 {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "odd-sample median and p95",
          message: "expected median 20 ns and nearest-rank p95 30 ns"))
      }
      let even = try P044LatencyStatistics.summarize([40, 10, 30, 20], percentile: 0.75)
      if even.medianNanoseconds == 25, even.percentileNanoseconds == 30 {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "even-sample median and p75",
          message: "expected median 25 ns and nearest-rank p75 30 ns"))
      }
    } catch {
      failures.append(JudgeFailure(caseName: "statistics", message: error.localizedDescription))
    }

    passed += expectError(name: "reject empty statistics", failures: &failures) {
      _ = try P044LatencyStatistics.summarize([])
    }
    passed += expectError(name: "reject invalid percentile", failures: &failures) {
      _ = try P044LatencyStatistics.summarize([1], percentile: 1.1)
    }

    do {
      let request = PrefillDecodeProfilingRequest(
        model: try EducationalMiniModelFixture.make(layerCount: 1),
        promptTokenIDs: [1, 4],
        decodeContextLengths: [2, 4],
        warmupIterations: 1,
        measuredTrials: 2,
        decodeStepsPerTrial: 2,
        percentile: 0.95,
        seed: 44)
      let report = try implementation(request)
      try P044ProfilingContract.validate(report, for: request)
      if report.prefill.stageName == "prefill.ttft",
        report.decode.allSatisfy({ $0.stageName.hasPrefix("decode.context.") })
      {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "profile shared mini-engine",
          message: "profile must preserve the named prefill and decode stages"))
      }
    } catch {
      failures.append(JudgeFailure(
        caseName: "profile shared mini-engine", message: error.localizedDescription))
    }

    passed += expectError(name: "reject zero measured trials", failures: &failures) {
      let request = PrefillDecodeProfilingRequest(
        model: try EducationalMiniModelFixture.make(layerCount: 1),
        promptTokenIDs: [1],
        decodeContextLengths: [1],
        warmupIterations: 0,
        measuredTrials: 0,
        decodeStepsPerTrial: 1)
      _ = try implementation(request)
    }

    return JudgeReport(passedCaseCount: passed, totalCaseCount: 6, failures: failures)
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