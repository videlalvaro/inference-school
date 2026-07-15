import Foundation
import InferenceSchoolCore
import InferenceSchoolRunnerProtocol

public struct RuntimeActivityDescriptor: Equatable, Sendable {
    public let id: String
    public let lessonID: String
    public let supportedStages: [RunStageID]
    public let exerciseFiles: [String]
    public let metalFile: String?

    public init(
        id: String,
        lessonID: String,
        supportedStages: [RunStageID],
        exerciseFiles: [String],
        metalFile: String?
    ) {
        self.id = id
        self.lessonID = lessonID
        self.supportedStages = supportedStages
        self.exerciseFiles = exerciseFiles
        self.metalFile = metalFile
    }
}

public enum RuntimeCheckExecutor {
    public static func events(for request: RunRequest) -> [RunEvent] {
        guard request.schemaVersion == RunnerProtocolVersion.current else {
            return rejectedEvents(
                for: request,
                code: "unsupported-schema-version",
                message: "Runner protocol version \(request.schemaVersion) is not supported."
            )
        }
        guard request.workspace == nil else {
            return rejectedEvents(
                for: request,
                code: "workspace-not-supported",
                message: "Workspace-backed checks require the toolchain runner."
            )
        }
        guard !request.stages.isEmpty else {
            return rejectedEvents(
                for: request,
                code: "missing-stage",
                message: "At least one check stage is required."
            )
        }
        guard Set(request.stages).count == request.stages.count else {
            return rejectedEvents(
                for: request,
                code: "duplicate-stage",
                message: "Each check stage may be requested only once."
            )
        }
        guard let activity = RuntimeRegistry.activity(forLessonID: request.lessonID),
              let runner = RuntimeRegistry.runner(for: request.lessonID)
        else {
            return rejectedEvents(
                for: request,
                code: "unknown-lesson",
                message: "No runtime activity is registered for lesson '\(request.lessonID)'."
            )
        }
        guard request.stages.allSatisfy(activity.supportedStages.contains) else {
            return rejectedEvents(
                for: request,
                code: "unsupported-stage",
                message: "The requested check stage is not available for lesson '\(request.lessonID)'."
            )
        }

        var events: [RunEvent] = []
        func emit(_ payload: RunEventPayload) {
            events.append(RunEvent(
                runID: request.runID,
                sequence: events.count,
                lessonID: request.lessonID,
                activityID: request.activityID,
                event: payload
            ))
        }

        emit(.accepted(RunAccepted(
            stageIDs: request.stages,
            implementation: request.implementation,
            mode: request.mode
        )))

        var passedStageCount = 0
        for stageID in request.stages {
            emit(.stageStarted(RunStage(id: stageID)))
            do {
                let report: JudgeReport
                switch stageID {
                case .cpu:
                    report = runner.cpuCheck(request.implementation == .canonical)
                case .metal:
                    guard let metalCheck = runner.metalCheck else {
                        preconditionFailure("Registry descriptor and runner stages disagree.")
                    }
                    report = try metalCheck(request.implementation == .canonical)
                }
                let structuredReport = structuredReport(from: report, stageID: stageID)
                emit(.judgeReport(structuredReport))
                if structuredReport.isPassing {
                    passedStageCount += 1
                }
            } catch {
                emit(.diagnostic(RunnerDiagnostic(
                    severity: .error,
                    code: "stage-setup-failed",
                    message: error.localizedDescription,
                    stageID: stageID
                )))
            }
        }

        emit(.completed(RunCompletion(
            status: passedStageCount == request.stages.count ? .passed : .failed,
            passedStageCount: passedStageCount,
            totalStageCount: request.stages.count
        )))
        return events
    }

    private static func rejectedEvents(
        for request: RunRequest,
        code: String,
        message: String
    ) -> [RunEvent] {
        [
            RunEvent(
                runID: request.runID,
                sequence: 0,
                lessonID: request.lessonID,
                activityID: request.activityID,
                event: .diagnostic(RunnerDiagnostic(
                    severity: .error,
                    code: code,
                    message: message
                ))
            ),
            RunEvent(
                runID: request.runID,
                sequence: 1,
                lessonID: request.lessonID,
                activityID: request.activityID,
                event: .completed(RunCompletion(
                    status: .rejected,
                    passedStageCount: 0,
                    totalStageCount: request.stages.count
                ))
            ),
        ]
    }

    private static func structuredReport(
        from report: JudgeReport,
        stageID: RunStageID
    ) -> RunJudgeReport {
        var occurrences: [String: Int] = [:]
        let failures = report.failures.map { failure in
            let baseID = caseID(from: failure.caseName)
            let occurrence = occurrences[baseID, default: 0] + 1
            occurrences[baseID] = occurrence
            let suffix = occurrence == 1 ? "" : "-\(occurrence)"
            return RunCaseFailure(
                id: "\(stageID.rawValue).\(baseID)\(suffix)",
                name: failure.caseName,
                message: failure.message
            )
        }
        return RunJudgeReport(
            stageID: stageID,
            passedCaseCount: report.passedCaseCount,
            totalCaseCount: report.totalCaseCount,
            failures: failures
        )
    }

    private static func caseID(from name: String) -> String {
        let components = name.lowercased().split { character in
            !character.isLetter && !character.isNumber
        }
        let identifier = components.joined(separator: "-")
        return identifier.isEmpty ? "case" : identifier
    }
}

public extension RuntimeRegistry {
    static func activity(forLessonID lessonID: String) -> RuntimeActivityDescriptor? {
        guard let runner = runner(for: lessonID) else {
            return nil
        }
        var stages: [RunStageID] = [.cpu]
        if runner.metalCheck != nil {
            stages.append(.metal)
        }
        return RuntimeActivityDescriptor(
            id: "\(lessonID).check",
            lessonID: lessonID,
            supportedStages: stages,
            exerciseFiles: runner.exerciseFiles,
            metalFile: runner.metalFile
        )
    }
}