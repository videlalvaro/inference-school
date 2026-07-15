import Darwin
import Foundation
import InferenceSchoolRunnerProtocol

public enum LocalRunnerClientError: Error, LocalizedError, Equatable {
    case executableMissing
    case launchFailed(String)
    case invalidEvent(String)
    case mismatchedEventIdentity
    case missingCompletion
    case processFailed(exitCode: Int32, message: String)

    public var errorDescription: String? {
        switch self {
        case .executableMissing:
            "The inference-school-runner executable could not be found."
        case let .launchFailed(message):
            "The runner could not launch: \(message)"
        case let .invalidEvent(message):
            "The runner emitted invalid JSONL: \(message)"
        case .mismatchedEventIdentity:
            "A runner event does not match the active request."
        case .missingCompletion:
            "The runner exited without a completion event."
        case let .processFailed(exitCode, message):
            "The runner exited with status \(exitCode): \(message)"
        }
    }
}

public struct LocalRunnerClient: Sendable {
    public let executableURL: URL

    public init(executableURL: URL) throws {
        let executableURL = executableURL.standardizedFileURL
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
            throw LocalRunnerClientError.executableMissing
        }
        self.executableURL = executableURL
    }

    public static func locate(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        applicationExecutableURL: URL? = Bundle.main.executableURL,
        applicationBundleURL: URL = Bundle.main.bundleURL
    ) throws -> Self {
        var candidates = [
            applicationBundleURL.appending(path: "Contents/Helpers/inference-school-runner")
        ]
        if let applicationExecutableURL {
            candidates.append(
                applicationExecutableURL.deletingLastPathComponent()
                    .appending(path: "inference-school-runner")
            )
        }
        if let configuredPath = environment["INFERENCE_SCHOOL_RUNNER_PATH"], !configuredPath.isEmpty {
            candidates.append(URL(fileURLWithPath: configuredPath))
        }
        guard let executableURL = candidates.first(where: {
            FileManager.default.isExecutableFile(atPath: $0.path)
        }) else {
            throw LocalRunnerClientError.executableMissing
        }
        return try Self(executableURL: executableURL)
    }

    public func events(
        for request: RunRequest
    ) -> AsyncThrowingStream<RunEvent, Error> {
        AsyncThrowingStream { continuation in
            let execution = RunnerExecution(
                executableURL: executableURL,
                request: request,
                continuation: continuation
            )
            continuation.onTermination = { @Sendable _ in
                execution.cancel()
            }
            execution.start()
        }
    }
}

private final class RunnerExecution: @unchecked Sendable {
    private let executableURL: URL
    private let request: RunRequest
    private let continuation: AsyncThrowingStream<RunEvent, Error>.Continuation
    private let process = Process()
    private let lock = NSLock()
    private var isCancelled = false
    private var failure: Error?
    private var receivedCompletion = false
    private var stderr = Data()
    private let maximumStderrBytes = 65_536

    init(
        executableURL: URL,
        request: RunRequest,
        continuation: AsyncThrowingStream<RunEvent, Error>.Continuation
    ) {
        self.executableURL = executableURL
        self.request = request
        self.continuation = continuation
    }

    func start() {
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            run()
        }
    }

    func cancel() {
        lock.withLock {
            isCancelled = true
            if process.isRunning {
                Self.terminateProcessTree(rootProcessID: process.processIdentifier)
            }
        }
    }

    private static func terminateProcessTree(rootProcessID: pid_t) {
        let descendants = descendantProcessIDs(of: rootProcessID)
        for processID in descendants.reversed() {
            kill(processID, SIGKILL)
        }
        if kill(-rootProcessID, SIGTERM) != 0 {
            kill(rootProcessID, SIGTERM)
        }
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

    private func run() {
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.executableURL = executableURL
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        do {
            let requestLine = try RunnerJSONL.encode(request) + "\n"
            try lock.withLock {
                guard !isCancelled else { throw CancellationError() }
                do {
                    try process.run()
                } catch {
                    throw LocalRunnerClientError.launchFailed(error.localizedDescription)
                }
            }
            try inputPipe.fileHandleForWriting.write(contentsOf: Data(requestLine.utf8))
            try inputPipe.fileHandleForWriting.close()
        } catch {
            continuation.finish(throwing: error)
            return
        }

        let readers = DispatchGroup()
        readers.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            readEvents(from: outputPipe.fileHandleForReading)
            readers.leave()
        }
        readers.enter()
        DispatchQueue.global(qos: .utility).async { [self] in
            readStderr(from: errorPipe.fileHandleForReading)
            readers.leave()
        }

        process.waitUntilExit()
        readers.wait()
        finish(exitCode: process.terminationStatus)
    }

    private func readEvents(from handle: FileHandle) {
        var pending = Data()
        while true {
            let data = handle.availableData
            guard !data.isEmpty else { break }
            pending.append(data)
            while let newline = pending.firstIndex(of: 0x0A) {
                let line = pending[..<newline]
                pending.removeSubrange(...newline)
                consume(line)
            }
        }
        if !pending.isEmpty {
            consume(pending[...])
        }
    }

    private func consume(_ data: Data.SubSequence) {
        guard failure == nil, !data.isEmpty else { return }
        do {
            guard let line = String(data: data, encoding: .utf8) else {
                throw LocalRunnerClientError.invalidEvent("Event data is not UTF-8.")
            }
            let event = try RunnerJSONL.decode(RunEvent.self, from: line)
            guard event.schemaVersion == RunnerProtocolVersion.current,
                  event.runID == request.runID,
                  event.lessonID == request.lessonID,
                  event.activityID == request.activityID
            else {
                throw LocalRunnerClientError.mismatchedEventIdentity
            }
            if case .completed = event.event {
                receivedCompletion = true
            }
            continuation.yield(event)
        } catch let error as LocalRunnerClientError {
            failure = error
            cancel()
        } catch {
            failure = LocalRunnerClientError.invalidEvent(error.localizedDescription)
            cancel()
        }
    }

    private func readStderr(from handle: FileHandle) {
        while true {
            let data = handle.availableData
            guard !data.isEmpty else { return }
            lock.withLock {
                let availableBytes = max(0, maximumStderrBytes - stderr.count)
                stderr.append(data.prefix(availableBytes))
            }
        }
    }

    private func finish(exitCode: Int32) {
        let state = lock.withLock {
            (
                isCancelled,
                failure,
                receivedCompletion,
                String(decoding: stderr, as: UTF8.self)
            )
        }
        if let failure = state.1 {
            continuation.finish(throwing: failure)
        } else if state.0 {
            continuation.finish(throwing: CancellationError())
        } else if exitCode != 0 {
            continuation.finish(throwing: LocalRunnerClientError.processFailed(
                exitCode: exitCode,
                message: state.3.trimmingCharacters(in: .whitespacesAndNewlines)
            ))
        } else if !state.2 {
            continuation.finish(throwing: LocalRunnerClientError.missingCompletion)
        } else {
            continuation.finish()
        }
    }
}