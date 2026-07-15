import Foundation
import InferenceSchoolWorkspaceKit
import XCTest

final class LearnerWorkspaceTests: XCTestCase {
    func testPreparePreservesLearnerEditsAndResetRestoresCourseSource() throws {
        try withTemporaryDirectory { destinationRoot in
            let sourceRoot = repositoryRoot
            let workspaceRoot = destinationRoot.appending(path: "workspace")
            let relativePath = "Sources/InferenceSchoolExercises/P001VectorDotExercise.swift"

            let workspace = try LearnerWorkspace.prepare(
                sourceRoot: sourceRoot,
                workspaceRoot: workspaceRoot
            )
            let original = try workspace.read(relativePath)
            try workspace.write("// learner edit\n", to: relativePath)

            let reopened = try LearnerWorkspace.prepare(
                sourceRoot: sourceRoot,
                workspaceRoot: workspaceRoot
            )
            XCTAssertEqual(try reopened.read(relativePath), "// learner edit\n")
            XCTAssertEqual(try reopened.reset(relativePath), original)
            XCTAssertEqual(try reopened.read(relativePath), original)
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: workspaceRoot.appending(path: "Package.swift").path
            ))
        }
    }

    func testFindsRepositoryRootFromLessonURL() throws {
        let lessonURL = repositoryRoot.appending(path: "Problems/001-vector-dot/README.md")

        XCTAssertEqual(
            LearnerWorkspace.findSourceRoot(containing: lessonURL),
            repositoryRoot.standardizedFileURL
        )
    }

    func testRejectsPathsOutsideWorkspace() throws {
        try withTemporaryDirectory { destinationRoot in
            let workspace = try LearnerWorkspace.prepare(
                sourceRoot: repositoryRoot,
                workspaceRoot: destinationRoot.appending(path: "workspace")
            )

            XCTAssertThrowsError(try workspace.fileURL(for: "../secret")) { error in
                XCTAssertEqual(
                    error as? LearnerWorkspaceError,
                    .unsafeRelativePath("../secret")
                )
            }
            XCTAssertThrowsError(try workspace.fileURL(for: "/tmp/secret"))
        }
    }

    func testPreparedWorkspaceBuildsCLIWithoutExternalPackages() throws {
        try withTemporaryDirectory { destinationRoot in
            let workspace = try LearnerWorkspace.prepare(
                sourceRoot: repositoryRoot,
                workspaceRoot: destinationRoot.appending(path: "workspace")
            )
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
            process.arguments = [
                "swift", "build",
                "--package-path", workspace.rootURL.path,
                "--product", "inference-school",
            ]
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice

            try process.run()
            process.waitUntilExit()

            XCTAssertEqual(process.terminationStatus, 0)
        }
    }

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func withTemporaryDirectory(_ operation: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString,
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try operation(root)
    }
}