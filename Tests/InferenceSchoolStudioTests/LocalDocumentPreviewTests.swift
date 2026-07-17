import Foundation
@testable import InferenceSchoolStudio
import XCTest

final class LocalDocumentPreviewTests: XCTestCase {
    func testResolvesMarkdownAndSourceDocuments() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let markdownContents = #"""
            # Math Primer

            $$
            \mathbf{x} = [1, 2, 3]
            $$
            """#
        let markdownURL = directory.appending(path: "primer.md")
        try markdownContents.write(to: markdownURL, atomically: true, encoding: .utf8)
        let markdown = try XCTUnwrap(PreviewedLocalDocument(resolving: markdownURL))
        XCTAssertEqual(markdown.title, "Math Primer")
        XCTAssertEqual(markdown.contents, markdownContents)
        XCTAssertEqual(
            markdown.renderedContents,
            "# Math Primer\n\n$$\\mathbf{x} = [1, 2, 3]$$"
        )
        XCTAssertEqual(markdown.kind, .markdown)

        let swiftURL = directory.appending(path: "Example.swift")
        try "let value = 42\n".write(to: swiftURL, atomically: true, encoding: .utf8)
        let swift = try XCTUnwrap(PreviewedLocalDocument(resolving: swiftURL))
        XCTAssertEqual(swift.title, "Example.swift")
        XCTAssertEqual(swift.renderedContents, swift.contents)
        XCTAssertEqual(swift.kind, .source(language: "swift"))

        let metalURL = directory.appending(path: "Kernel.metal")
        try "kernel void example() {}\n".write(to: metalURL, atomically: true, encoding: .utf8)
        let metal = try XCTUnwrap(PreviewedLocalDocument(resolving: metalURL))
        XCTAssertEqual(metal.title, "Kernel.metal")
        XCTAssertEqual(metal.kind, .source(language: "metal"))
    }

    func testRejectsExternalUnsupportedAndMissingDocuments() throws {
        XCTAssertNil(PreviewedLocalDocument(resolving: try XCTUnwrap(
            URL(string: "https://example.com/guide.md")
        )))

        let directory = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let unsupportedURL = directory.appending(path: "notes.txt")
        try "notes".write(to: unsupportedURL, atomically: true, encoding: .utf8)
        XCTAssertNil(PreviewedLocalDocument(resolving: unsupportedURL))
        XCTAssertNil(PreviewedLocalDocument(resolving: directory.appending(path: "missing.swift")))
    }

    func testEveryProblemLocalLinkTargetsAPreviewableFile() throws {
        let problemsRoot = repositoryRoot.appending(path: "Problems", directoryHint: .isDirectory)
        let lessonURLs = try FileManager.default.contentsOfDirectory(
            at: problemsRoot,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        .map { $0.appending(path: "README.md") }
        .filter { FileManager.default.fileExists(atPath: $0.path) }
        .sorted { $0.path < $1.path }
        XCTAssertEqual(lessonURLs.count, 48)

        var linkCount = 0
        for lessonURL in lessonURLs {
            let markdown = try String(contentsOf: lessonURL, encoding: .utf8)
            let attributed = try AttributedString(
                markdown: markdown,
                baseURL: lessonURL.deletingLastPathComponent()
            )
            for targetURL in attributed.runs.compactMap(\.link) where targetURL.isFileURL {
                XCTAssertTrue(
                    FileManager.default.fileExists(atPath: targetURL.path),
                    "Missing target from \(lessonURL.path): \(targetURL.path)"
                )
                XCTAssertNotNil(
                    PreviewedLocalDocument(resolving: targetURL),
                    "Unsupported target from \(lessonURL.path): \(targetURL.path)"
                )
                linkCount += 1
            }
        }
        XCTAssertEqual(linkCount, 191)
    }

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
