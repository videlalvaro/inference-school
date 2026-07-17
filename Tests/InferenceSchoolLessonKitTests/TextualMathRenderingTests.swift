import Foundation
import InferenceSchoolLessonKit
import Textual
import XCTest

final class TextualMathRenderingTests: XCTestCase {
    @MainActor
    func testMultilineVectorDisplayBecomesMathAttachment() throws {
        let markdown = #"""
            $$
            \mathbf{x} = [1, -2, 3, 4], \qquad
            \mathbf{y} = [0.5, 2, -1, 0.25],
            $$
            """#
        let parser = AttributedStringMarkdownParser(
            baseURL: nil,
            syntaxExtensions: [.math]
        )

        let normalized = LessonMarkdownRendering.normalizeDisplayMath(in: markdown)
        let rendered = try parser.attributedString(for: normalized)

        XCTAssertEqual(normalized.components(separatedBy: "\n").count, 1)
        XCTAssertTrue(normalized.hasPrefix("$$"))
        XCTAssertTrue(normalized.hasSuffix("$$"))
        XCTAssertFalse(String(rendered.characters).contains("$$"))
        XCTAssertTrue(String(rendered.characters).contains("\u{FFFC}"))
    }

    @MainActor
    func testUnsupportedCourseMathCommandsAreNormalized() throws {
        let markdown = #"""
            $$\operatorname{SiLU}(x) = x \bmod 4$$

            Inline $\boldsymbol{\gamma}$ remains math.
            """#
        let parser = AttributedStringMarkdownParser(
            baseURL: nil,
            syntaxExtensions: [.math]
        )

        let normalized = LessonMarkdownRendering.normalizeDisplayMath(in: markdown)
        let rendered = try parser.attributedString(for: normalized)

        XCTAssertFalse(normalized.contains(#"\operatorname"#))
        XCTAssertFalse(normalized.contains(#"\bmod"#))
        XCTAssertFalse(normalized.contains(#"\boldsymbol"#))
        XCTAssertTrue(normalized.contains(#"\mathrm{SiLU}"#))
        XCTAssertTrue(normalized.contains(#"\mathrm{mod}"#))
        XCTAssertTrue(normalized.contains(#"\mathbf{\gamma}"#))
        XCTAssertEqual(String(rendered.characters).filter { $0 == "\u{FFFC}" }.count, 2)
    }

    func testDisplayMathInsideCodeFenceIsNotNormalized() {
        let markdown = #"""
            ```text
            $$
            x + y
            $$
            ```
            """#

        XCTAssertEqual(
            LessonMarkdownRendering.normalizeDisplayMath(in: markdown),
            markdown
        )
    }

    func testUnsupportedMathCommandsInProseAndInlineCodeArePreserved() {
        let markdown = #"""
            Prose keeps \operatorname{SiLU}, \bmod, and \boldsymbol{gamma}.

            Inline code keeps `\operatorname{SiLU}`, `\bmod`, and `\boldsymbol{gamma}`.

            Math changes $\operatorname{SiLU}(x) = x \bmod 4$ and $\boldsymbol{\gamma}$.
            """#

        XCTAssertEqual(
            LessonMarkdownRendering.normalizeDisplayMath(in: markdown),
            #"""
            Prose keeps \operatorname{SiLU}, \bmod, and \boldsymbol{gamma}.

            Inline code keeps `\operatorname{SiLU}`, `\bmod`, and `\boldsymbol{gamma}`.

            Math changes $\mathrm{SiLU}(x) = x \mathrm{mod} 4$ and $\mathbf{\gamma}$.
            """#
        )
    }

    func testMultilineInlineCodeIsPreserved() {
        let markdown = #"""
            Code starts `here
            $$
            \operatorname{SiLU}(x) = x \bmod 4
            $$
            and ends here`.

            Math changes $\boldsymbol{\gamma}$.
            """#

        XCTAssertEqual(
            LessonMarkdownRendering.normalizeDisplayMath(in: markdown),
            #"""
            Code starts `here
            $$
            \operatorname{SiLU}(x) = x \bmod 4
            $$
            and ends here`.

            Math changes $\mathbf{\gamma}$.
            """#
        )
    }

    func testUnmatchedBacktickDoesNotSuppressLaterMath() {
        let markdown = #"""
            An unmatched ` stays literal.

            Math changes $\operatorname{SiLU}(x)$.
            """#

        XCTAssertEqual(
            LessonMarkdownRendering.normalizeDisplayMath(in: markdown),
            #"""
            An unmatched ` stays literal.

            Math changes $\mathrm{SiLU}(x)$.
            """#
        )
    }

    func testShorterBacktickRunDoesNotCloseLongerFence() {
        let markdown = #"""
            ````text
            ```
            $$
            \operatorname{SiLU}(x) = x \bmod 4
            $$
            ```
            ````
            """#

        XCTAssertEqual(
            LessonMarkdownRendering.normalizeDisplayMath(in: markdown),
            markdown
        )
    }

    func testShorterTildeRunDoesNotCloseLongerFence() {
        let markdown = #"""
            ~~~~text
            ~~~
            $$
            \boldsymbol{\gamma}
            $$
            ~~~
            ~~~~
            """#

        XCTAssertEqual(
            LessonMarkdownRendering.normalizeDisplayMath(in: markdown),
            markdown
        )
    }

    func testCompletionChecklistBecomesTypedBlock() {
        let lesson = LessonDocument(
            id: "sample",
            title: "Sample",
            order: 1,
            sourceURL: URL(fileURLWithPath: "/sample.md"),
            relativePath: "sample.md",
            markdown: """
            # Sample

            Before the checklist.

            ## Completion checklist

            - [ ] Run the CPU judge.
            - [x] Explain the `Float32` result.

            ## Next section

            After the checklist.
            """
        )

        XCTAssertEqual(
            LessonMarkdownRendering.blocks(in: lesson),
            [
                .markdown("# Sample\n\nBefore the checklist.\n"),
                .checklist(
                    anchor: "completion-checklist",
                    title: "Completion checklist",
                    items: [
                        LessonChecklistItem(
                            id: "completion-1",
                            text: "Run the CPU judge.",
                            isCompleted: false
                        ),
                        LessonChecklistItem(
                            id: "completion-2",
                            text: "Explain the `Float32` result.",
                            isCompleted: true
                        ),
                    ]
                ),
                .markdown("## Next section\n\nAfter the checklist."),
            ]
        )
    }
}