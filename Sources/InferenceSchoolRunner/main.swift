import Darwin
import Foundation
import InferenceSchoolRuntime
import InferenceSchoolRunnerProtocol

guard setpgid(0, 0) == 0 || getpgrp() == getpid() else {
    FileHandle.standardError.write(Data("runner process-group setup failed\n".utf8))
    exit(2)
}

private func write(_ event: RunEvent) throws {
    let line = try RunnerJSONL.encode(event) + "\n"
    try FileHandle.standardOutput.write(contentsOf: Data(line.utf8))
}

private func write(_ events: [RunEvent]) throws {
    try events.forEach(write)
}

private func rejectedEvents(for error: Error) -> [RunEvent] {
    let request = RunRequest(
        runID: UUID().uuidString,
        lessonID: "unknown",
        activityID: "unknown",
        stages: []
    )
    return [
        RunEvent(
            runID: request.runID,
            sequence: 0,
            lessonID: request.lessonID,
            activityID: request.activityID,
            event: .diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "invalid-request",
                message: error.localizedDescription
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
                totalStageCount: 0
            ))
        ),
    ]
}

while let line = readLine(strippingNewline: true) {
    guard !line.isEmpty else {
        continue
    }
    do {
        let request = try RunnerJSONL.decode(RunRequest.self, from: line)
        if request.workspace == nil {
            try write(RuntimeCheckExecutor.events(for: request))
        } else {
            var outputError: Error?
            _ = WorkspaceCheckRunner.events(for: request) { event in
                guard outputError == nil else { return }
                do {
                    try write(event)
                } catch {
                    outputError = error
                }
            }
            if let outputError {
                throw outputError
            }
        }
    } catch {
        do {
            try write(rejectedEvents(for: error))
        } catch {
            FileHandle.standardError.write(Data("runner output failure: \(error)\n".utf8))
            exit(2)
        }
    }
}