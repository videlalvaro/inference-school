import InferenceSchoolCore
import InferenceSchoolRuntime
import InferenceSchoolLessonKit
import InferenceSchoolRunnerProtocol
import XCTest

final class RuntimeRegistryTests: XCTestCase {
    func testEveryCourseProblemHasExactlyOneRuntimeAdapter() {
        let problemIDs = Course.availableProblems.map(\.id)
        let missingIDs = problemIDs.filter { RuntimeRegistry.runner(for: $0) == nil }

        XCTAssertEqual(problemIDs.count, 47)
        XCTAssertEqual(Set(problemIDs).count, problemIDs.count)
        XCTAssertTrue(missingIDs.isEmpty, "Missing runtime adapters: \(missingIDs)")
        XCTAssertNil(RuntimeRegistry.runner(for: "048"))
    }

    func testMarkdownCatalogRuntimeAndLearnerSourcesDoNotDrift() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let lessons = try LessonCatalog.discover(in: repositoryRoot)

        XCTAssertEqual(lessons.count, 48)
        let orientation = try XCTUnwrap(lessons.first)
        XCTAssertEqual(orientation.id, "000")
        XCTAssertNil(RuntimeRegistry.activity(forLessonID: orientation.id))

        let runnableLessons = Array(lessons.dropFirst())
        XCTAssertEqual(runnableLessons.map(\.id), Course.availableProblems.map(\.id))
        for lesson in runnableLessons {
            let activity = try XCTUnwrap(
                RuntimeRegistry.activity(forLessonID: lesson.id),
                "Missing runtime activity for Markdown lesson \(lesson.id)."
            )
            XCTAssertEqual(activity.id, "\(lesson.id).check")
            for sourcePath in activity.exerciseFiles + [activity.metalFile].compactMap({ $0 }) {
                XCTAssertTrue(
                    FileManager.default.fileExists(
                        atPath: repositoryRoot.appending(path: sourcePath).path
                    ),
                    "Runtime source does not exist: \(sourcePath)"
                )
            }
        }
    }

    func testCanonicalP001RunsThroughExportedRegistry() throws {
        let runner = try XCTUnwrap(RuntimeRegistry.runner(for: "001"))

        let cpuReport = runner.cpuCheck(true)
        XCTAssertTrue(cpuReport.isPassing)
        XCTAssertGreaterThan(cpuReport.totalCaseCount, 0)

        let metalReport = try XCTUnwrap(runner.metalCheck)(true)
        XCTAssertTrue(metalReport.isPassing)
        XCTAssertGreaterThan(metalReport.totalCaseCount, 0)
    }

    func testRegistryExportsLearnerSourceLocations() throws {
        let runner = try XCTUnwrap(RuntimeRegistry.runner(for: "001"))

        XCTAssertEqual(
            runner.exerciseFiles,
            ["Sources/InferenceSchoolExercises/P001VectorDotExercise.swift"]
        )
        XCTAssertEqual(
            runner.metalFile,
            "Sources/InferenceSchoolExercises/Metal/P001VectorDot.metal"
        )
    }

    func testCanonicalCheckEmitsVersionedOrderedEvents() throws {
        let request = RunRequest(
            runID: "run-p001-cpu",
            lessonID: "001",
            activityID: "001.check",
            stages: [.cpu],
            implementation: .canonical
        )

        let events = RuntimeCheckExecutor.events(for: request)

        XCTAssertEqual(events.map(\.schemaVersion), [1, 1, 1, 1])
        XCTAssertEqual(events.map(\.runID), Array(repeating: "run-p001-cpu", count: 4))
        XCTAssertEqual(events.map(\.sequence), [0, 1, 2, 3])
        guard case let .judgeReport(report) = events[2].event else {
            return XCTFail("Expected a judge report as the third event.")
        }
        XCTAssertEqual(report.stageID, .cpu)
        XCTAssertEqual(report.passedCaseCount, 5)
        XCTAssertEqual(report.totalCaseCount, 5)
        XCTAssertTrue(report.failures.isEmpty)
        guard case let .completed(completion) = events[3].event else {
            return XCTFail("Expected a completion as the final event.")
        }
        XCTAssertEqual(completion.status, .passed)
    }

    func testJSONLEventRoundTripsWithTaggedPayload() throws {
        let request = RunRequest(
            runID: "jsonl-round-trip",
            lessonID: "001",
            activityID: "001.check",
            stages: [.cpu],
            implementation: .canonical
        )
        let event = try XCTUnwrap(RuntimeCheckExecutor.events(for: request).first)

        let line = try RunnerJSONL.encode(event)
        let decoded = try RunnerJSONL.decode(RunEvent.self, from: line)

        XCTAssertEqual(decoded, event)
        XCTAssertFalse(line.contains("\n"))
        XCTAssertTrue(line.contains(#""schemaVersion":1"#))
        XCTAssertTrue(line.contains(#""type":"accepted""#))
    }

    func testUnsupportedProtocolVersionIsRejectedWithoutRunningJudge() {
        let request = RunRequest(
            schemaVersion: 999,
            runID: "future-schema",
            lessonID: "001",
            activityID: "001.check",
            stages: [.cpu],
            implementation: .canonical
        )

        let events = RuntimeCheckExecutor.events(for: request)

        XCTAssertEqual(events.count, 2)
        guard case let .diagnostic(diagnostic) = events[0].event else {
            return XCTFail("Expected a rejection diagnostic.")
        }
        XCTAssertEqual(diagnostic.code, "unsupported-schema-version")
        guard case let .completed(completion) = events[1].event else {
            return XCTFail("Expected a rejected completion.")
        }
        XCTAssertEqual(completion.status, .rejected)
    }
}