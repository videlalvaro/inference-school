import Foundation
import InferenceSchoolLessonKit
import XCTest

final class LessonCatalogTests: XCTestCase {
    func testDiscoversTheBuiltInCourseWithoutTheSwiftRegistry() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let lessons = try LessonCatalog.discover(in: repositoryRoot)

        XCTAssertEqual(lessons.count, 48)
        XCTAssertEqual(lessons.first?.id, "000")
        XCTAssertEqual(lessons.last?.id, "047")
        XCTAssertEqual(
            lessons.first(where: { $0.id == "014" })?.title,
            "Q/K/V Projections and Head Views"
        )

        let orientation = try XCTUnwrap(lessons.first)
        XCTAssertEqual(orientation.title, "Start Here: Build an LLM Inference Engine")
        XCTAssertTrue(orientation.activities.allSatisfy { $0.kind == "mermaid" })
    }

    func testEveryBuiltInCompletionChecklistBecomesTypedContent() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let lessons = try LessonCatalog.discover(in: repositoryRoot)

        XCTAssertEqual(lessons.count, 48)
        for lesson in lessons {
            let blocks = LessonMarkdownRendering.blocks(in: lesson)
            XCTAssertEqual(
                blocks.filter {
                    if case .checklist = $0 { return true }
                    return false
                }.count,
                1,
                "\(lesson.id) must expose one typed completion checklist"
            )
            XCTAssertFalse(
                blocks.contains {
                    guard case let .markdown(markdown) = $0 else { return false }
                    return markdown.contains("## Completion checklist")
                        || markdown.contains("- [ ]")
                },
                "\(lesson.id) must not leave its checklist in generic Markdown"
            )
        }
    }

    func testPlainMarkdownUsesPathAndHeadingFallbacks() throws {
        try withTemporaryCourse { root in
            try write(
                "# A New Operator\n\nThis lesson needs no manifest.\n",
                to: root.appending(path: "Lessons/008-new-operator.md")
            )

            let lesson = try XCTUnwrap(LessonCatalog.discover(in: root).first)
            XCTAssertEqual(lesson.id, "Lessons.008-new-operator.md")
            XCTAssertEqual(lesson.title, "A New Operator")
            XCTAssertEqual(lesson.summary, "This lesson needs no manifest.")
            XCTAssertTrue(lesson.activities.isEmpty)
        }
    }

    func testFrontMatterAndInteractiveFencesAreParsed() throws {
        try withTemporaryCourse { root in
            try write(
                """
                ---
                formatVersion: 1
                id: sample.qkv
                title: Head Mapping
                order: 14
                module: attention
                prerequisites: [sample.tensor]
                tags: [qkv, shapes]
                capabilities: [swift]
                contentVersion: 3
                ---
                # Ignored Fallback Title

                ```quiz {#head-count}
                schemaVersion: 1
                prompt: How many query heads?
                ```

                ```swift {#head-index .exercise evaluator="swift-snippet"}
                func headIndex() -> Int { 0 }
                ```
                """,
                to: root.appending(path: "head-mapping.md")
            )

            let lesson = try XCTUnwrap(LessonCatalog.discover(in: root).first)
            XCTAssertEqual(lesson.id, "sample.qkv")
            XCTAssertEqual(lesson.title, "Head Mapping")
            XCTAssertEqual(lesson.order, 14)
            XCTAssertEqual(lesson.module, "attention")
            XCTAssertEqual(lesson.prerequisites, ["sample.tensor"])
            XCTAssertEqual(lesson.tags, ["qkv", "shapes"])
            XCTAssertEqual(lesson.contentVersion, 3)
            XCTAssertEqual(lesson.activities.map(\.id), ["head-count", "head-index"])
            XCTAssertEqual(lesson.activities.map(\.kind), ["quiz", "swift"])
            XCTAssertEqual(lesson.sections.map(\.title), ["Ignored Fallback Title"])
            XCTAssertTrue(lesson.diagnostics.isEmpty)
        }
    }

    func testMermaidDirectivesBecomeOrderedRenderBlocks() throws {
        try withTemporaryCourse { root in
            try write(
                """
                # Reduction

                Before the diagram.

                ```mermaid {#reduction-tree}
                flowchart LR
                    A --> B
                ```

                After the diagram.
                """,
                to: root.appending(path: "reduction.md")
            )

            let lesson = try XCTUnwrap(LessonCatalog.discover(in: root).first)
            XCTAssertEqual(lesson.activities.map(\.id), ["reduction-tree"])
            XCTAssertEqual(lesson.activities.map(\.kind), ["mermaid"])
            XCTAssertEqual(
                LessonMarkdownRendering.blocks(in: lesson),
                [
                    .markdown("# Reduction\n\nBefore the diagram.\n"),
                    .mermaid(id: "reduction-tree", source: "flowchart LR\n    A --> B"),
                    .markdown("\nAfter the diagram."),
                ]
            )
        }
    }

    func testMultipleRootsReportDuplicateLessonIDsWithoutHidingEitherLesson() throws {
        try withTemporaryCourse { root in
            let firstRoot = root.appending(path: "first", directoryHint: .isDirectory)
            let secondRoot = root.appending(path: "second", directoryHint: .isDirectory)
            let lesson = "---\nid: shared.lesson\n---\n# Shared\n"
            try write(lesson, to: firstRoot.appending(path: "one.md"))
            try write(lesson, to: secondRoot.appending(path: "two.md"))

            let snapshot = try LessonCatalog.load(from: [
                LessonContentRoot(id: "first", url: firstRoot),
                LessonContentRoot(id: "second", url: secondRoot),
            ])

            XCTAssertEqual(snapshot.lessons.count, 2)
            XCTAssertEqual(snapshot.diagnostics.map(\.code), ["duplicate-lesson-id"])
            XCTAssertEqual(snapshot.diagnostics.first?.sourceURLs.count, 2)
            XCTAssertEqual(snapshot.revisionHash.count, 64)
        }
    }

    func testUnsupportedAndUnknownDirectivesRemainReadableWithDiagnostics() throws {
        try withTemporaryCourse { root in
            try write(
                """
                # Extensible Lesson

                ```quiz {#future-quiz}
                schemaVersion: 99
                prompt: Future format
                ```

                ```hologram {#view}
                schemaVersion: 1
                source: tensor.json
                ```
                """,
                to: root.appending(path: "extensible.md")
            )

            let lesson = try XCTUnwrap(LessonCatalog.discover(in: root).first)
            XCTAssertTrue(lesson.activities.isEmpty)
            XCTAssertEqual(
                lesson.diagnostics.map(\.code),
                ["unsupported-directive-version", "unknown-directive"]
            )
            XCTAssertTrue(lesson.markdown.contains("Future format"))
            XCTAssertTrue(lesson.markdown.contains("tensor.json"))
        }
    }

    func testUpdateStreamEmitsForAddEditAndDelete() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let sourceURL = root.appending(path: "live.md")
        var updates = LessonCatalog.updates(
            from: [LessonContentRoot(id: "live", url: root)],
            every: .milliseconds(10)
        ).makeAsyncIterator()

        let initialUpdate = await updates.next()
        let initial = try XCTUnwrap(initialUpdate)
        XCTAssertTrue(initial.lessons.isEmpty)

        try write("# First Title\n", to: sourceURL)
        let addedUpdate = await updates.next()
        let added = try XCTUnwrap(addedUpdate)
        XCTAssertEqual(added.lessons.map(\.title), ["First Title"])

        try write("# Revised Title\n", to: sourceURL)
        let editedUpdate = await updates.next()
        let edited = try XCTUnwrap(editedUpdate)
        XCTAssertEqual(edited.lessons.map(\.title), ["Revised Title"])

        try FileManager.default.removeItem(at: sourceURL)
        let deletedUpdate = await updates.next()
        let deleted = try XCTUnwrap(deletedUpdate)
        XCTAssertTrue(deleted.lessons.isEmpty)
    }

    func testMalformedFrontMatterDoesNotHideTheLesson() throws {
        try withTemporaryCourse { root in
            try write(
                "---\ntags: [unterminated\n---\n# Still Visible\n",
                to: root.appending(path: "broken.md")
            )

            let lesson = try XCTUnwrap(LessonCatalog.discover(in: root).first)
            XCTAssertEqual(lesson.title, "Still Visible")
            XCTAssertEqual(lesson.diagnostics.first?.severity, .error)
        }
    }

    private func withTemporaryCourse(_ operation: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try operation(root)
    }

    private func write(_ contents: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }
}