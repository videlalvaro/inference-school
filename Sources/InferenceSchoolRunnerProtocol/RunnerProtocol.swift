import Foundation

public enum RunnerProtocolVersion {
    public static let current = 1
}

public enum RunMode: String, Codable, Sendable {
    case debug
    case release
}

public enum RunImplementation: String, Codable, Sendable {
    case learner
    case canonical
}

public enum RunStageID: String, Codable, CaseIterable, Sendable {
    case cpu
    case metal
}

public struct RunLimits: Codable, Equatable, Sendable {
    public let timeoutMilliseconds: Int?
    public let maximumOutputBytes: Int?
    public let maximumArtifactBytes: Int?

    public init(
        timeoutMilliseconds: Int? = nil,
        maximumOutputBytes: Int? = nil,
        maximumArtifactBytes: Int? = nil
    ) {
        self.timeoutMilliseconds = timeoutMilliseconds
        self.maximumOutputBytes = maximumOutputBytes
        self.maximumArtifactBytes = maximumArtifactBytes
    }
}

public struct RunRequest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let runID: String
    public let lessonID: String
    public let activityID: String
    public let workspace: String?
    public let workspaceBookmark: Data?
    public let mode: RunMode
    public let toolchain: String?
    public let limits: RunLimits
    public let stages: [RunStageID]
    public let implementation: RunImplementation

    public init(
        schemaVersion: Int = RunnerProtocolVersion.current,
        runID: String = UUID().uuidString,
        lessonID: String,
        activityID: String,
        workspace: String? = nil,
        workspaceBookmark: Data? = nil,
        mode: RunMode = .debug,
        toolchain: String? = nil,
        limits: RunLimits = RunLimits(),
        stages: [RunStageID],
        implementation: RunImplementation = .learner
    ) {
        self.schemaVersion = schemaVersion
        self.runID = runID
        self.lessonID = lessonID
        self.activityID = activityID
        self.workspace = workspace
        self.workspaceBookmark = workspaceBookmark
        self.mode = mode
        self.toolchain = toolchain
        self.limits = limits
        self.stages = stages
        self.implementation = implementation
    }
}

public enum RunnerDiagnosticSeverity: String, Codable, Sendable {
    case note
    case warning
    case error
}

public struct RunnerDiagnostic: Codable, Equatable, Sendable {
    public let severity: RunnerDiagnosticSeverity
    public let code: String
    public let message: String
    public let stageID: RunStageID?
    public let caseID: String?

    public init(
        severity: RunnerDiagnosticSeverity,
        code: String,
        message: String,
        stageID: RunStageID? = nil,
        caseID: String? = nil
    ) {
        self.severity = severity
        self.code = code
        self.message = message
        self.stageID = stageID
        self.caseID = caseID
    }
}

public struct RunAccepted: Codable, Equatable, Sendable {
    public let stageIDs: [RunStageID]
    public let implementation: RunImplementation
    public let mode: RunMode

    public init(stageIDs: [RunStageID], implementation: RunImplementation, mode: RunMode) {
        self.stageIDs = stageIDs
        self.implementation = implementation
        self.mode = mode
    }
}

public struct RunStage: Codable, Equatable, Sendable {
    public let id: RunStageID

    public init(id: RunStageID) {
        self.id = id
    }
}

public struct RunBuild: Codable, Equatable, Sendable {
    public let workspace: String
    public let mode: RunMode
    public let toolchain: String?

    public init(workspace: String, mode: RunMode, toolchain: String? = nil) {
        self.workspace = workspace
        self.mode = mode
        self.toolchain = toolchain
    }
}

public struct RunBuildResult: Codable, Equatable, Sendable {
    public let succeeded: Bool
    public let exitCode: Int32

    public init(succeeded: Bool, exitCode: Int32) {
        self.succeeded = succeeded
        self.exitCode = exitCode
    }
}

public struct RunOutput: Codable, Equatable, Sendable {
    public let text: String
    public let truncated: Bool

    public init(text: String, truncated: Bool = false) {
        self.text = text
        self.truncated = truncated
    }
}

public struct RunCaseFailure: Codable, Equatable, Sendable {
    public let id: String
    public let name: String
    public let message: String

    public init(id: String, name: String, message: String) {
        self.id = id
        self.name = name
        self.message = message
    }
}

public struct RunJudgeReport: Codable, Equatable, Sendable {
    public let stageID: RunStageID
    public let passedCaseCount: Int
    public let totalCaseCount: Int
    public let failures: [RunCaseFailure]

    public var isPassing: Bool {
        failures.isEmpty && passedCaseCount == totalCaseCount
    }

    public init(
        stageID: RunStageID,
        passedCaseCount: Int,
        totalCaseCount: Int,
        failures: [RunCaseFailure]
    ) {
        self.stageID = stageID
        self.passedCaseCount = passedCaseCount
        self.totalCaseCount = totalCaseCount
        self.failures = failures
    }
}

public enum RunCompletionStatus: String, Codable, Sendable {
    case passed
    case failed
    case rejected
    case cancelled
    case timedOut
    case crashed
}

public struct RunCompletion: Codable, Equatable, Sendable {
    public let status: RunCompletionStatus
    public let passedStageCount: Int
    public let totalStageCount: Int

    public init(status: RunCompletionStatus, passedStageCount: Int, totalStageCount: Int) {
        self.status = status
        self.passedStageCount = passedStageCount
        self.totalStageCount = totalStageCount
    }
}

public enum RunEventPayload: Equatable, Sendable {
    case accepted(RunAccepted)
    case buildStarted(RunBuild)
    case buildFinished(RunBuildResult)
    case stageStarted(RunStage)
    case judgeReport(RunJudgeReport)
    case diagnostic(RunnerDiagnostic)
    case stdout(RunOutput)
    case stderr(RunOutput)
    case completed(RunCompletion)
}

extension RunEventPayload: Codable {
    private enum Kind: String, Codable {
        case accepted
        case buildStarted
        case buildFinished
        case stageStarted
        case judgeReport
        case diagnostic
        case stdout
        case stderr
        case completed
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case payload
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .type) {
        case .accepted:
            self = .accepted(try container.decode(RunAccepted.self, forKey: .payload))
        case .buildStarted:
            self = .buildStarted(try container.decode(RunBuild.self, forKey: .payload))
        case .buildFinished:
            self = .buildFinished(try container.decode(RunBuildResult.self, forKey: .payload))
        case .stageStarted:
            self = .stageStarted(try container.decode(RunStage.self, forKey: .payload))
        case .judgeReport:
            self = .judgeReport(try container.decode(RunJudgeReport.self, forKey: .payload))
        case .diagnostic:
            self = .diagnostic(try container.decode(RunnerDiagnostic.self, forKey: .payload))
        case .stdout:
            self = .stdout(try container.decode(RunOutput.self, forKey: .payload))
        case .stderr:
            self = .stderr(try container.decode(RunOutput.self, forKey: .payload))
        case .completed:
            self = .completed(try container.decode(RunCompletion.self, forKey: .payload))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .accepted(payload):
            try container.encode(Kind.accepted, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .buildStarted(payload):
            try container.encode(Kind.buildStarted, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .buildFinished(payload):
            try container.encode(Kind.buildFinished, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .stageStarted(payload):
            try container.encode(Kind.stageStarted, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .judgeReport(payload):
            try container.encode(Kind.judgeReport, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .diagnostic(payload):
            try container.encode(Kind.diagnostic, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .stdout(payload):
            try container.encode(Kind.stdout, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .stderr(payload):
            try container.encode(Kind.stderr, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case let .completed(payload):
            try container.encode(Kind.completed, forKey: .type)
            try container.encode(payload, forKey: .payload)
        }
    }
}

public struct RunEvent: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let runID: String
    public let sequence: Int
    public let lessonID: String
    public let activityID: String
    public let event: RunEventPayload

    public init(
        schemaVersion: Int = RunnerProtocolVersion.current,
        runID: String,
        sequence: Int,
        lessonID: String,
        activityID: String,
        event: RunEventPayload
    ) {
        self.schemaVersion = schemaVersion
        self.runID = runID
        self.sequence = sequence
        self.lessonID = lessonID
        self.activityID = activityID
        self.event = event
    }
}

public enum RunnerJSONL {
    public static func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        guard let line = String(data: data, encoding: .utf8) else {
            throw EncodingError.invalidValue(
                value,
                EncodingError.Context(codingPath: [], debugDescription: "JSON is not UTF-8.")
            )
        }
        return line
    }

    public static func decode<T: Decodable>(_ type: T.Type, from line: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(line.utf8))
    }
}