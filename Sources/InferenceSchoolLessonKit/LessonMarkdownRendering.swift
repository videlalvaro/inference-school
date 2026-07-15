import Foundation

public struct LessonChecklistItem: Identifiable, Hashable, Sendable {
    public let id: String
    public let text: String
    public let isCompleted: Bool

    public init(id: String, text: String, isCompleted: Bool) {
        self.id = id
        self.text = text
        self.isCompleted = isCompleted
    }
}

public enum LessonContentBlock: Hashable, Sendable {
    case markdown(String)
    case mermaid(id: String, source: String)
    case checklist(anchor: String, title: String, items: [LessonChecklistItem])
}

public enum LessonMarkdownRendering {
    public static func blocks(in lesson: LessonDocument) -> [LessonContentBlock] {
        let diagrams = lesson.activities
            .filter { $0.kind == "mermaid" }
            .sorted { $0.sourceLines.lowerBound < $1.sourceLines.lowerBound }
        let lines = lesson.markdown
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var blocks: [LessonContentBlock] = []
        var nextLineIndex = 0

        for diagram in diagrams {
            let openingLineIndex = max(0, diagram.sourceLines.lowerBound - 1)
            appendMarkdown(
                lines[nextLineIndex..<min(openingLineIndex, lines.count)],
                to: &blocks
            )
            blocks.append(.mermaid(id: diagram.id, source: diagram.configuration))
            nextLineIndex = min(diagram.sourceLines.upperBound, lines.count)
        }
        appendMarkdown(lines[nextLineIndex..<lines.count], to: &blocks)
        return blocks.flatMap(expandCompletionChecklist)
    }

    public static func normalizeDisplayMath(in markdown: String) -> String {
        let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false)
        var output: [String] = []
        var mathLines: [String]?
        var openingDelimiter = ""
        var fence: Character?

        for lineSlice in lines {
            let line = String(lineSlice)
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if mathLines == nil, let marker = fenceMarker(in: trimmed) {
                fence = fence == nil ? marker : (fence == marker ? nil : fence)
                output.append(line)
                continue
            }

            guard fence == nil, trimmed == "$$" else {
                if mathLines != nil {
                    mathLines?.append(textualCompatibleMath(in: trimmed))
                } else {
                    output.append(fence == nil ? textualCompatibleMath(in: line) : line)
                }
                continue
            }

            if let capturedLines = mathLines {
                let latex = capturedLines.filter { !$0.isEmpty }.joined(separator: " ")
                output.append("\(openingDelimiter)$$\(latex)$$")
                mathLines = nil
                openingDelimiter = ""
            } else {
                openingDelimiter = String(line.prefix { $0.isWhitespace })
                mathLines = []
            }
        }

        if let mathLines {
            output.append("\(openingDelimiter)$$")
            output.append(contentsOf: mathLines)
        }

        return output.joined(separator: "\n")
    }

    private static func textualCompatibleMath(in line: String) -> String {
        line
            .replacingOccurrences(
                of: #"\\operatorname\{([^{}]+)\}"#,
                with: #"\\mathrm{$1}"#,
                options: .regularExpression
            )
            .replacingOccurrences(of: #"\bmod"#, with: #"\mathrm{mod}"#)
            .replacingOccurrences(of: #"\boldsymbol"#, with: #"\mathbf"#)
    }

    private static func fenceMarker(in line: String) -> Character? {
        guard let marker = line.first, marker == "`" || marker == "~" else { return nil }
        return line.prefix { $0 == marker }.count >= 3 ? marker : nil
    }

    private static func appendMarkdown(
        _ lines: ArraySlice<String>,
        to blocks: inout [LessonContentBlock]
    ) {
        let markdown = lines.joined(separator: "\n")
        guard !markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        blocks.append(.markdown(normalizeDisplayMath(in: markdown)))
    }

    private static func expandCompletionChecklist(
        _ block: LessonContentBlock
    ) -> [LessonContentBlock] {
        guard case let .markdown(markdown) = block else { return [block] }
        let lines = markdown
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var blocks: [LessonContentBlock] = []
        var markdownStart = 0
        var lineIndex = 0

        while lineIndex < lines.count {
            guard let title = completionChecklistTitle(in: lines[lineIndex]) else {
                lineIndex += 1
                continue
            }

            var sectionEnd = lineIndex + 1
            while sectionEnd < lines.count, !isHeading(lines[sectionEnd]) {
                sectionEnd += 1
            }
            let sectionLines = lines[(lineIndex + 1)..<sectionEnd]
            let nonemptyLines = sectionLines.filter {
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            let parsedItems = nonemptyLines.enumerated().compactMap { index, line in
                checklistItem(in: line, index: index)
            }
            guard !parsedItems.isEmpty, parsedItems.count == nonemptyLines.count else {
                lineIndex = sectionEnd
                continue
            }

            appendMarkdown(lines[markdownStart..<lineIndex], to: &blocks)
            blocks.append(.checklist(
                anchor: "completion-checklist",
                title: title,
                items: parsedItems
            ))
            markdownStart = sectionEnd
            lineIndex = sectionEnd
        }

        appendMarkdown(lines[markdownStart..<lines.count], to: &blocks)
        return blocks.isEmpty ? [block] : blocks
    }

    private static func completionChecklistTitle(in line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let marker = trimmed.prefix { $0 == "#" }
        guard (1...6).contains(marker.count),
            trimmed.dropFirst(marker.count).first == " "
        else { return nil }
        let title = trimmed.dropFirst(marker.count + 1)
            .trimmingCharacters(in: CharacterSet(charactersIn: " #\t"))
        return title.localizedCaseInsensitiveCompare("Completion checklist") == .orderedSame
            ? title
            : nil
    }

    private static func isHeading(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let marker = trimmed.prefix { $0 == "#" }
        return (1...6).contains(marker.count)
            && trimmed.dropFirst(marker.count).first == " "
    }

    private static func checklistItem(in line: String, index: Int) -> LessonChecklistItem? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard let bullet = trimmed.first, "-*+".contains(bullet) else { return nil }
        let task = trimmed.dropFirst().trimmingCharacters(in: .whitespaces)
        guard task.count >= 3, task.first == "[", task.dropFirst(2).first == "]" else {
            return nil
        }
        let marker = task[task.index(after: task.startIndex)]
        guard marker == " " || marker == "x" || marker == "X" else { return nil }
        let text = task.dropFirst(3).trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return nil }
        return LessonChecklistItem(
            id: "completion-\(index + 1)",
            text: text,
            isCompleted: marker == "x" || marker == "X"
        )
    }
}