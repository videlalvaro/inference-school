import Foundation
import InferenceSchoolRunnerClient
import InferenceSchoolRunnerProtocol
import InferenceSchoolWorkspaceKit
import XCTest

final class RunnerPipelineTests: XCTestCase {
    private static let environmentSentinel = "INFERENCE_SCHOOL_TEST_PARENT_CREDENTIAL"

    func testLocatePrefersBundledRunnerOverEnvironmentOverride() async throws {
        try await withTemporaryDirectory { destinationRoot in
            let bundleURL = destinationRoot.appending(path: "Inference School Studio.app")
            let bundledRunnerURL = bundleURL.appending(
                path: "Contents/Helpers/inference-school-runner"
            )
            let overrideRunnerURL = destinationRoot.appending(path: "override-runner")
            try makeExecutable(at: bundledRunnerURL)
            try makeExecutable(at: overrideRunnerURL)

            let client = try LocalRunnerClient.locate(
                environment: ["INFERENCE_SCHOOL_RUNNER_PATH": overrideRunnerURL.path],
                applicationExecutableURL: bundleURL.appending(
                    path: "Contents/MacOS/inference-school-studio"
                ),
                applicationBundleURL: bundleURL
            )

            XCTAssertEqual(client.executableURL, bundledRunnerURL.standardizedFileURL)
        }
    }

    func testLocateUsesEnvironmentOverrideWithoutBundledRunner() async throws {
        try await withTemporaryDirectory { destinationRoot in
            let overrideRunnerURL = destinationRoot.appending(path: "override-runner")
            try makeExecutable(at: overrideRunnerURL)

            let client = try LocalRunnerClient.locate(
                environment: ["INFERENCE_SCHOOL_RUNNER_PATH": overrideRunnerURL.path],
                applicationExecutableURL: nil,
                applicationBundleURL: destinationRoot.appending(path: "Missing.app")
            )

            XCTAssertEqual(client.executableURL, overrideRunnerURL.standardizedFileURL)
        }
    }

    func testRealRunnerBuildsAndChecksIsolatedP001Starter() async throws {
        guard let runnerPath = ProcessInfo.processInfo.environment["INFERENCE_SCHOOL_RUNNER_PATH"] else {
            throw XCTSkip("Set INFERENCE_SCHOOL_RUNNER_PATH to run the process integration test.")
        }
        try await withTemporaryDirectory { destinationRoot in
            let workspace = try LearnerWorkspace.prepare(
                sourceRoot: repositoryRoot,
                workspaceRoot: destinationRoot.appending(path: "workspace")
            )
            let client = try LocalRunnerClient(
                executableURL: URL(fileURLWithPath: runnerPath)
            )
            let request = RunRequest(
                runID: "p001-learner-integration",
                lessonID: "001",
                activityID: "001.check",
                workspace: workspace.rootURL.path,
                limits: RunLimits(
                    timeoutMilliseconds: 60_000,
                    maximumOutputBytes: 1_048_576
                ),
                stages: [.cpu]
            )

            var events: [RunEvent] = []
            let clock = ContinuousClock()
            let startedAt = clock.now
            var firstEventElapsed: Duration?
            for try await event in client.events(for: request) {
                if firstEventElapsed == nil {
                    firstEventElapsed = startedAt.duration(to: clock.now)
                }
                events.append(event)
            }

            XCTAssertLessThan(try XCTUnwrap(firstEventElapsed), .seconds(2))
            let buildResult = try XCTUnwrap(events.compactMap { event -> RunBuildResult? in
                guard case let .buildFinished(result) = event.event else { return nil }
                return result
            }.last)
            XCTAssertTrue(buildResult.succeeded)

            let report = try XCTUnwrap(events.compactMap { event -> RunJudgeReport? in
                guard case let .judgeReport(report) = event.event else { return nil }
                return report
            }.last)
            XCTAssertEqual(report.stageID, .cpu)
            XCTAssertEqual(report.totalCaseCount, 5)
            XCTAssertFalse(report.isPassing)
            XCTAssertFalse(report.failures.isEmpty)

            let completion = try XCTUnwrap(events.compactMap { event -> RunCompletion? in
                guard case let .completed(completion) = event.event else { return nil }
                return completion
            }.last)
            XCTAssertEqual(completion.status, .failed)
            XCTAssertEqual(events.map(\.sequence), Array(events.indices))
            XCTAssertTrue(events.allSatisfy { $0.runID == request.runID })
        }
    }

    func testRunnerDoesNotForwardUnrelatedParentEnvironment() async throws {
        guard let runnerPath = ProcessInfo.processInfo.environment["INFERENCE_SCHOOL_RUNNER_PATH"] else {
            throw XCTSkip("Set INFERENCE_SCHOOL_RUNNER_PATH to run the process integration test.")
        }
        try await withTemporaryDirectory { destinationRoot in
            let workspace = try LearnerWorkspace.prepare(
                sourceRoot: repositoryRoot,
                workspaceRoot: destinationRoot.appending(path: "workspace")
            )
            try environmentCheckingP001Source.write(
                to: workspace.fileURL(
                    for: "Sources/InferenceSchoolExercises/P001VectorDotExercise.swift"
                ),
                atomically: true,
                encoding: .utf8
            )

            let runnerShim = destinationRoot.appending(path: "runner-with-sentinel")
            let shim = """
                #!/bin/zsh
                export \(Self.environmentSentinel)=test-only-value
                exec \(shellSingleQuoted(runnerPath))
                """
            try shim.write(to: runnerShim, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o700],
                ofItemAtPath: runnerShim.path
            )

            let client = try LocalRunnerClient(executableURL: runnerShim)
            let request = RunRequest(
                runID: "p001-environment-isolation",
                lessonID: "001",
                activityID: "001.check",
                workspace: workspace.rootURL.path,
                limits: RunLimits(
                    timeoutMilliseconds: 60_000,
                    maximumOutputBytes: 1_048_576
                ),
                stages: [.cpu]
            )

            var report: RunJudgeReport?
            for try await event in client.events(for: request) {
                if case let .judgeReport(value) = event.event {
                    report = value
                }
            }

            XCTAssertTrue(try XCTUnwrap(report).isPassing)
        }
    }

    func testCancellationStopsWorkspaceBuildProcessTree() async throws {
        guard let runnerPath = ProcessInfo.processInfo.environment["INFERENCE_SCHOOL_RUNNER_PATH"] else {
            throw XCTSkip("Set INFERENCE_SCHOOL_RUNNER_PATH to run the process integration test.")
        }
        try await withTemporaryDirectory { destinationRoot in
            let workspace = try LearnerWorkspace.prepare(
                sourceRoot: repositoryRoot,
                workspaceRoot: destinationRoot.appending(path: "workspace")
            )
            let client = try LocalRunnerClient(
                executableURL: URL(fileURLWithPath: runnerPath)
            )
            let request = RunRequest(
                runID: "p001-cancellation-integration",
                lessonID: "001",
                activityID: "001.check",
                workspace: workspace.rootURL.path,
                limits: RunLimits(timeoutMilliseconds: 60_000),
                stages: [.cpu]
            )
            let buildStarted = expectation(description: "runner started workspace build")
            let runTask = Task {
                do {
                    for try await event in client.events(for: request) {
                        if case .buildStarted = event.event {
                            buildStarted.fulfill()
                        }
                    }
                } catch is CancellationError {
                    return
                } catch {
                    XCTFail("Unexpected runner error: \(error)")
                }
            }

            await fulfillment(of: [buildStarted], timeout: 2)
            let buildProcessAppeared = try await waitUntil(timeout: .seconds(2)) {
                try !processCommands(containing: workspace.rootURL.path).isEmpty
            }
            XCTAssertTrue(buildProcessAppeared)

            runTask.cancel()
            await runTask.value

            let buildProcessStopped = try await waitUntil(timeout: .seconds(3)) {
                try processCommands(containing: workspace.rootURL.path).isEmpty
            }
            XCTAssertTrue(buildProcessStopped)
        }
    }

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var environmentCheckingP001Source: String {
        """
        import Foundation
        import InferenceSchoolCore

        public enum P001VectorDotExercise {
            public static func dot(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
                guard lhs.count == rhs.count else {
                    throw VectorDotError.lengthMismatch(lhs: lhs.count, rhs: rhs.count)
                }
                guard ProcessInfo.processInfo.environment["\(Self.environmentSentinel)"] == nil else {
                    return .nan
                }
                return Float(zip(lhs, rhs).reduce(0.0) { result, pair in
                    result + Double(pair.0) * Double(pair.1)
                })
            }
        }
        """
    }

    private func shellSingleQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    private func makeExecutable(at url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try "#!/bin/sh\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: url.path
        )
    }

    private func withTemporaryDirectory(
        _ operation: (URL) async throws -> Void
    ) async throws {
        let root = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString,
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try await operation(root)
    }

    private func waitUntil(
        timeout: Duration,
        condition: () throws -> Bool
    ) async throws -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        repeat {
            if try condition() {
                return true
            }
            try await Task.sleep(for: .milliseconds(50))
        } while clock.now < deadline
        return try condition()
    }

    private func processCommands(containing text: String) throws -> [String] {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-axo", "command="]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return [] }
        return String(decoding: data, as: UTF8.self)
            .split(separator: "\n")
            .map(String.init)
            .filter { $0.contains(text) }
    }
}