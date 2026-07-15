import Darwin
import Foundation
import InferenceSchoolRuntime
import InferenceSchoolRunnerProtocol

enum WorkspaceCheckRunner {
    static func events(
        for request: RunRequest,
        onEvent: ((RunEvent) -> Void)? = nil
    ) -> [RunEvent] {
        var events: [RunEvent] = []
        func emit(_ payload: RunEventPayload) {
            let event = RunEvent(
                runID: request.runID,
                sequence: events.count,
                lessonID: request.lessonID,
                activityID: request.activityID,
                event: payload
            )
            events.append(event)
            onEvent?(event)
        }
        func complete(_ status: RunCompletionStatus, passedStages: Int = 0) {
            emit(.completed(RunCompletion(
                status: status,
                passedStageCount: passedStages,
                totalStageCount: request.stages.count
            )))
        }
        func reject(code: String, message: String) -> [RunEvent] {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: code,
                message: message
            )))
            complete(.rejected)
            return events
        }

        guard request.schemaVersion == RunnerProtocolVersion.current else {
            return reject(
                code: "unsupported-schema-version",
                message: "Runner protocol version \(request.schemaVersion) is not supported."
            )
        }
        guard let workspace = request.workspace, !workspace.isEmpty else {
            return reject(code: "missing-workspace", message: "A workspace path is required.")
        }
        let workspaceURL = URL(fileURLWithPath: workspace, isDirectory: true)
            .standardizedFileURL
        let workspaceAccess: SecurityScopedWorkspaceAccess?
        do {
            workspaceAccess = try SecurityScopedWorkspaceAccess.acquire(
                bookmarkData: request.workspaceBookmark,
                workspaceURL: workspaceURL
            )
        } catch {
            return reject(
                code: "workspace-authorization-failed",
                message: error.localizedDescription
            )
        }
        defer { workspaceAccess?.stop() }
        guard FileManager.default.fileExists(
            atPath: workspaceURL.appending(path: "Package.swift").path
        ) else {
            return reject(
                code: "invalid-workspace",
                message: "The workspace does not contain Package.swift."
            )
        }
        guard !request.stages.isEmpty, Set(request.stages).count == request.stages.count else {
            return reject(
                code: "invalid-stages",
                message: "At least one unique check stage is required."
            )
        }
        guard let activity = RuntimeRegistry.activity(forLessonID: request.lessonID) else {
            return reject(
                code: "unknown-lesson",
                message: "No runtime activity is registered for lesson '\(request.lessonID)'."
            )
        }
        guard request.stages.allSatisfy(activity.supportedStages.contains) else {
            return reject(
                code: "unsupported-stage",
                message: "The requested stage is not available for lesson '\(request.lessonID)'."
            )
        }
        if let timeout = request.limits.timeoutMilliseconds, timeout <= 0 {
            return reject(code: "invalid-timeout", message: "The timeout must be positive.")
        }
        if let maximumOutputBytes = request.limits.maximumOutputBytes,
           maximumOutputBytes <= 0
        {
            return reject(
                code: "invalid-output-limit",
                message: "The output limit must be positive."
            )
        }

        emit(.accepted(RunAccepted(
            stageIDs: request.stages,
            implementation: request.implementation,
            mode: request.mode
        )))
        emit(.buildStarted(RunBuild(
            workspace: workspaceURL.path,
            mode: request.mode,
            toolchain: request.toolchain
        )))

        let maximumOutputBytes = request.limits.maximumOutputBytes ?? 1_048_576
        let build: SwiftCourseBuild
        do {
            build = try SwiftCourseBuilder.build(
                workspaceURL: workspaceURL,
                mode: request.mode,
                environment: deterministicEnvironment(),
                timeoutMilliseconds: request.limits.timeoutMilliseconds,
                maximumOutputBytes: maximumOutputBytes
            )
        } catch {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "build-launch-failed",
                message: error.localizedDescription
            )))
            emit(.buildFinished(RunBuildResult(succeeded: false, exitCode: -1)))
            complete(.failed)
            return events
        }
        let buildResult = build.result
        emitOutput(from: buildResult, emit: emit)
        emit(.buildFinished(RunBuildResult(
            succeeded: buildResult.exitCode == 0 && !buildResult.timedOut
                && !buildResult.outputLimitExceeded,
            exitCode: buildResult.exitCode
        )))
        if buildResult.timedOut {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "build-timed-out",
                message: "The workspace build exceeded its time limit."
            )))
            complete(.timedOut)
            return events
        }
        if buildResult.outputLimitExceeded {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "output-limit-exceeded",
                message: "Build output exceeded \(maximumOutputBytes) bytes."
            )))
            complete(.failed)
            return events
        }
        guard buildResult.exitCode == 0 else {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "build-failed",
                message: "The Swift compiler exited with status \(buildResult.exitCode)."
            )))
            complete(.failed)
            return events
        }

        let executableURL = build.executableURL
            .standardizedFileURL
            .resolvingSymlinksInPath()
        let resolvedWorkspaceURL = workspaceURL.resolvingSymlinksInPath()
        guard executableURL.path.hasPrefix(resolvedWorkspaceURL.path + "/"),
              FileManager.default.isExecutableFile(atPath: executableURL.path)
        else {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "binary-lookup-failed",
                message: "The Swift compiler did not produce a contained executable."
            )))
            complete(.failed)
            return events
        }

        var checkArguments = ["check", request.lessonID]
        if request.stages == [.cpu] {
            checkArguments.append("--cpu")
        } else if request.stages == [.metal] {
            checkArguments.append("--metal")
        }
        if request.implementation == .canonical {
            checkArguments.append("--solution")
        }
        checkArguments += [
            "--format", "jsonl",
            "--run-id", request.runID,
            "--activity-id", request.activityID,
        ]

        let checkResult: ChildProcessResult
        do {
            checkResult = try ChildProcess.run(
                executableURL: executableURL,
                arguments: checkArguments,
                environment: deterministicEnvironment(),
                timeoutMilliseconds: request.limits.timeoutMilliseconds,
                maximumOutputBytes: maximumOutputBytes
            )
        } catch {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "check-launch-failed",
                message: error.localizedDescription
            )))
            complete(.crashed)
            return events
        }
        if !checkResult.stderr.isEmpty {
            emit(.stderr(RunOutput(
                text: checkResult.stderr,
                truncated: checkResult.stderrTruncated
            )))
        }
        if checkResult.timedOut {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "check-timed-out",
                message: "The learner check exceeded its time limit."
            )))
            complete(.timedOut)
            return events
        }
        if checkResult.outputLimitExceeded {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "output-limit-exceeded",
                message: "Check output exceeded \(maximumOutputBytes) bytes."
            )))
            complete(.failed)
            return events
        }

        var childCompletion: RunCompletion?
        for line in checkResult.stdout.split(separator: "\n", omittingEmptySubsequences: true) {
            do {
                let childEvent = try RunnerJSONL.decode(RunEvent.self, from: String(line))
                guard childEvent.schemaVersion == RunnerProtocolVersion.current,
                      childEvent.runID == request.runID,
                      childEvent.lessonID == request.lessonID,
                      childEvent.activityID == request.activityID
                else {
                    throw WorkspaceRunnerError.mismatchedChildEvent
                }
                switch childEvent.event {
                case .accepted:
                    break
                case let .completed(completion):
                    childCompletion = completion
                default:
                    emit(childEvent.event)
                }
            } catch {
                emit(.stdout(RunOutput(text: String(line))))
                emit(.diagnostic(RunnerDiagnostic(
                    severity: .error,
                    code: "invalid-child-event",
                    message: error.localizedDescription
                )))
                complete(.crashed)
                return events
            }
        }
        guard let childCompletion else {
            emit(.diagnostic(RunnerDiagnostic(
                severity: .error,
                code: "missing-completion",
                message: "The learner check exited without a completion event."
            )))
            complete(checkResult.exitCode == 0 ? .crashed : .failed)
            return events
        }
        emit(.completed(childCompletion))
        return events
    }

    private static func emitOutput(
        from result: ChildProcessResult,
        emit: (RunEventPayload) -> Void
    ) {
        if !result.stdout.isEmpty {
            emit(.stdout(RunOutput(text: result.stdout, truncated: result.stdoutTruncated)))
        }
        if !result.stderr.isEmpty {
            emit(.stderr(RunOutput(text: result.stderr, truncated: result.stderrTruncated)))
        }
    }

    private static func deterministicEnvironment() -> [String: String] {
        let parent = ProcessInfo.processInfo.environment
        let inheritedKeys = [
            "HOME",
            "TMPDIR",
            "DEVELOPER_DIR",
            "SDKROOT",
            "TOOLCHAINS",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
        ]
        var environment = Dictionary(uniqueKeysWithValues: inheritedKeys.compactMap { key in
            parent[key].map { (key, $0) }
        })
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        environment["CLICOLOR"] = "0"
        environment["NO_COLOR"] = "1"
        environment["TERM"] = "dumb"
        return environment
    }
}

private enum WorkspaceRunnerError: Error, LocalizedError {
    case mismatchedChildEvent

    var errorDescription: String? {
        "The child event identity does not match its run request."
    }
}

struct ChildProcessResult {
    let exitCode: Int32
    let stdout: String
    let stderr: String
    let stdoutTruncated: Bool
    let stderrTruncated: Bool
    let outputLimitExceeded: Bool
    let timedOut: Bool
}

enum ChildProcess {
    static func run(
        executableURL: URL,
        arguments: [String],
        environment: [String: String],
        timeoutMilliseconds: Int?,
        maximumOutputBytes: Int
    ) throws -> ChildProcessResult {
        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let collector = BoundedOutputCollector(maximumBytes: maximumOutputBytes)
        let readers = DispatchGroup()
        let terminated = DispatchSemaphore(value: 0)

        process.executableURL = executableURL
        process.arguments = arguments
        process.environment = environment
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        process.terminationHandler = { _ in terminated.signal() }

        startReader(
            stdoutPipe.fileHandleForReading,
            stream: .stdout,
            collector: collector,
            group: readers
        )
        startReader(
            stderrPipe.fileHandleForReading,
            stream: .stderr,
            collector: collector,
            group: readers
        )

        try process.run()
        let timedOut: Bool
        if let timeoutMilliseconds {
            timedOut = terminated.wait(
                timeout: .now() + .milliseconds(timeoutMilliseconds)
            ) == .timedOut
        } else {
            terminated.wait()
            timedOut = false
        }

        if timedOut {
            terminateProcessTree(rootProcessID: process.processIdentifier)
            if terminated.wait(timeout: .now() + .milliseconds(500)) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                terminated.wait()
            }
        }
        readers.wait()
        let output = collector.snapshot()
        return ChildProcessResult(
            exitCode: process.terminationStatus,
            stdout: String(decoding: output.stdout, as: UTF8.self),
            stderr: String(decoding: output.stderr, as: UTF8.self),
            stdoutTruncated: output.stdoutTruncated,
            stderrTruncated: output.stderrTruncated,
            outputLimitExceeded: output.limitExceeded,
            timedOut: timedOut
        )
    }

    private static func terminateProcessTree(rootProcessID: pid_t) {
        let descendants = descendantProcessIDs(of: rootProcessID)
        for processID in descendants.reversed() {
            kill(processID, SIGKILL)
        }
        kill(rootProcessID, SIGTERM)
    }

    private static func descendantProcessIDs(of rootProcessID: pid_t) -> [pid_t] {
        var descendants: [pid_t] = []
        var pending = [rootProcessID]
        var visited = Set([rootProcessID])

        while let parentProcessID = pending.popLast() {
            for childProcessID in childProcessIDs(of: parentProcessID)
            where visited.insert(childProcessID).inserted {
                descendants.append(childProcessID)
                pending.append(childProcessID)
            }
        }
        return descendants
    }

    private static func childProcessIDs(of parentProcessID: pid_t) -> [pid_t] {
        let capacity = proc_listchildpids(parentProcessID, nil, 0)
        guard capacity > 0 else { return [] }

        var processIDs = [pid_t](repeating: 0, count: Int(capacity))
        let count = processIDs.withUnsafeMutableBytes { buffer in
            proc_listchildpids(parentProcessID, buffer.baseAddress, Int32(buffer.count))
        }
        guard count > 0 else { return [] }
        return processIDs.prefix(Int(count)).filter { $0 > 0 }
    }

    private static func startReader(
        _ handle: FileHandle,
        stream: BoundedOutputCollector.Stream,
        collector: BoundedOutputCollector,
        group: DispatchGroup
    ) {
        group.enter()
        DispatchQueue.global(qos: .utility).async {
            defer { group.leave() }
            while true {
                let data = handle.availableData
                guard !data.isEmpty else { return }
                collector.append(data, to: stream)
            }
        }
    }
}

private final class BoundedOutputCollector: @unchecked Sendable {
    enum Stream {
        case stdout
        case stderr
    }

    struct Snapshot {
        let stdout: Data
        let stderr: Data
        let stdoutTruncated: Bool
        let stderrTruncated: Bool
        let limitExceeded: Bool
    }

    private let maximumBytes: Int
    private let lock = NSLock()
    private var stdout = Data()
    private var stderr = Data()
    private var stdoutTruncated = false
    private var stderrTruncated = false
    private var totalBytes = 0

    init(maximumBytes: Int) {
        self.maximumBytes = maximumBytes
    }

    func append(_ data: Data, to stream: Stream) {
        lock.withLock {
            let availableBytes = max(0, maximumBytes - totalBytes)
            let acceptedBytes = min(availableBytes, data.count)
            if acceptedBytes > 0 {
                switch stream {
                case .stdout:
                    stdout.append(data.prefix(acceptedBytes))
                case .stderr:
                    stderr.append(data.prefix(acceptedBytes))
                }
                totalBytes += acceptedBytes
            }
            if acceptedBytes < data.count {
                switch stream {
                case .stdout:
                    stdoutTruncated = true
                case .stderr:
                    stderrTruncated = true
                }
            }
        }
    }

    func snapshot() -> Snapshot {
        lock.withLock {
            Snapshot(
                stdout: stdout,
                stderr: stderr,
                stdoutTruncated: stdoutTruncated,
                stderrTruncated: stderrTruncated,
                limitExceeded: stdoutTruncated || stderrTruncated
            )
        }
    }
}