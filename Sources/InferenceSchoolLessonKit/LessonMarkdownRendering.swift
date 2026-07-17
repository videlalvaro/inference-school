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
        let backtickRunLines = backtickRunLines(in: lines)
        let paragraphEndLines = paragraphEndLines(in: lines)
        var output: [String] = []
        var mathLines: [String]?
        var openingMathLine = ""
        var openingDelimiter = ""
        var fence: MarkdownFence?
        var codeSpanDelimiterLength: Int?

        for (lineIndex, lineSlice) in lines.enumerated() {
            let line = String(lineSlice)
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if let activeFence = fence {
                output.append(line)
                if closesFence(line, openedBy: activeFence) {
                    fence = nil
                }
                continue
            }

            if mathLines == nil, codeSpanDelimiterLength == nil,
               let openingFence = openingFence(in: line)
            {
                fence = openingFence
                output.append(line)
                continue
            }

            if var capturedLines = mathLines {
                if trimmed == "$$" {
                    let latex = capturedLines
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                        .joined(separator: " ")
                    output.append(
                        "\(openingDelimiter)$$\(textualCompatibleMath(in: latex))$$"
                    )
                    mathLines = nil
                    openingMathLine = ""
                    openingDelimiter = ""
                } else {
                    capturedLines.append(line)
                    mathLines = capturedLines
                }
                continue
            }

            if codeSpanDelimiterLength == nil, trimmed == "$$" {
                openingMathLine = line
                openingDelimiter = String(line.prefix { $0.isWhitespace })
                mathLines = []
            } else {
                output.append(normalizeInlineMath(
                    in: line,
                    lineIndex: lineIndex,
                    paragraphEndLine: paragraphEndLines[lineIndex],
                    backtickRunLines: backtickRunLines,
                    codeSpanDelimiterLength: &codeSpanDelimiterLength
                ))
            }
        }

        if let mathLines {
            output.append(openingMathLine)
            output.append(contentsOf: mathLines)
        }

        return output.joined(separator: "\n")
    }

    private static func normalizeInlineMath(
        in line: String,
        lineIndex: Int,
        paragraphEndLine: Int,
        backtickRunLines: [Int: [Int]],
        codeSpanDelimiterLength: inout Int?
    ) -> String {
        var output = ""
        var index = line.startIndex

        while index < line.endIndex {
            if let openingLength = codeSpanDelimiterLength {
                guard line[index] == "`" else {
                    output.append(line[index])
                    index = line.index(after: index)
                    continue
                }
                let delimiterEnd = endOfRun(of: "`", in: line, from: index)
                let delimiterLength = line.distance(from: index, to: delimiterEnd)
                output.append(contentsOf: line[index..<delimiterEnd])
                index = delimiterEnd
                if delimiterLength == openingLength {
                    codeSpanDelimiterLength = nil
                }
                continue
            }

            let marker = line[index]
            guard marker == "`" || marker == "$", !isEscaped(index, in: line) else {
                output.append(marker)
                index = line.index(after: index)
                continue
            }

            let delimiterEnd = endOfRun(of: marker, in: line, from: index)
            let delimiterLength = line.distance(from: index, to: delimiterEnd)
            if marker == "`" {
                if let closingDelimiter = closingDelimiter(
                    in: line,
                    marker: marker,
                    length: delimiterLength,
                    after: delimiterEnd
                ) {
                    output.append(contentsOf: line[index..<closingDelimiter.upperBound])
                    index = closingDelimiter.upperBound
                } else {
                    output.append(contentsOf: line[index..<delimiterEnd])
                    if hasCodeSpanClosingDelimiter(
                        length: delimiterLength,
                        afterLine: lineIndex,
                        beforeLine: paragraphEndLine,
                        backtickRunLines: backtickRunLines
                    ) {
                        codeSpanDelimiterLength = delimiterLength
                    }
                    index = delimiterEnd
                }
                continue
            }
            if marker == "$", delimiterLength > 2 {
                output.append(contentsOf: line[index..<delimiterEnd])
                index = delimiterEnd
                continue
            }

            guard let closingDelimiter = closingDelimiter(
                in: line,
                marker: marker,
                length: delimiterLength,
                after: delimiterEnd
            ) else {
                output.append(contentsOf: line[index..<delimiterEnd])
                index = delimiterEnd
                continue
            }

            output.append(contentsOf: line[index..<delimiterEnd])
            output.append(contentsOf: textualCompatibleMath(
                in: String(line[delimiterEnd..<closingDelimiter.lowerBound])
            ))
            output.append(contentsOf: line[closingDelimiter])
            index = closingDelimiter.upperBound
        }

        return output
    }

    private static func hasCodeSpanClosingDelimiter(
        length: Int,
        afterLine lineIndex: Int,
        beforeLine paragraphEndLine: Int,
        backtickRunLines: [Int: [Int]]
    ) -> Bool {
        guard let runLines = backtickRunLines[length] else { return false }
        var lowerBound = 0
        var upperBound = runLines.count
        while lowerBound < upperBound {
            let middle = lowerBound + (upperBound - lowerBound) / 2
            if runLines[middle] <= lineIndex {
                lowerBound = middle + 1
            } else {
                upperBound = middle
            }
        }
        return lowerBound < runLines.count && runLines[lowerBound] < paragraphEndLine
    }

    private static func backtickRunLines(in lines: [Substring]) -> [Int: [Int]] {
        var runLines: [Int: [Int]] = [:]
        for (lineIndex, lineSlice) in lines.enumerated() {
            let line = String(lineSlice)
            var index = line.startIndex
            while index < line.endIndex {
                guard line[index] == "`" else {
                    index = line.index(after: index)
                    continue
                }
                let delimiterEnd = endOfRun(of: "`", in: line, from: index)
                let delimiterLength = line.distance(from: index, to: delimiterEnd)
                runLines[delimiterLength, default: []].append(lineIndex)
                index = delimiterEnd
            }
        }
        return runLines
    }

    private static func paragraphEndLines(in lines: [Substring]) -> [Int] {
        var endLines = Array(repeating: lines.count, count: lines.count)
        var paragraphEndLine = lines.count
        for lineIndex in lines.indices.reversed() {
            let line = lines[lineIndex]
            if line.allSatisfy(\.isWhitespace) {
                paragraphEndLine = lineIndex
            }
            endLines[lineIndex] = paragraphEndLine
        }
        return endLines
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

    private struct MarkdownFence {
        let marker: Character
        let length: Int
    }

    private static func openingFence(in line: String) -> MarkdownFence? {
        let content = line.drop { $0.isWhitespace }
        guard let marker = content.first, marker == "`" || marker == "~" else {
            return nil
        }
        let markerLength = content.prefix { $0 == marker }.count
        guard markerLength >= 3 else { return nil }
        let remainder = content.dropFirst(markerLength)
        guard marker != "`" || !remainder.contains("`") else { return nil }
        return MarkdownFence(marker: marker, length: markerLength)
    }

    private static func closesFence(_ line: String, openedBy fence: MarkdownFence) -> Bool {
        let content = line.drop { $0.isWhitespace }
        let markerLength = content.prefix { $0 == fence.marker }.count
        guard markerLength >= fence.length else { return false }
        return content.dropFirst(markerLength).allSatisfy(\.isWhitespace)
    }

    private static func endOfRun(
        of marker: Character,
        in text: String,
        from startIndex: String.Index
    ) -> String.Index {
        var index = startIndex
        while index < text.endIndex, text[index] == marker {
            index = text.index(after: index)
        }
        return index
    }

    private static func closingDelimiter(
        in text: String,
        marker: Character,
        length: Int,
        after startIndex: String.Index
    ) -> Range<String.Index>? {
        var index = startIndex
        while index < text.endIndex {
            guard text[index] == marker else {
                index = text.index(after: index)
                continue
            }
            let delimiterEnd = endOfRun(of: marker, in: text, from: index)
            let delimiterLength = text.distance(from: index, to: delimiterEnd)
            if delimiterLength == length, marker == "`" || !isEscaped(index, in: text) {
                return index..<delimiterEnd
            }
            index = delimiterEnd
        }
        return nil
    }

    private static func isEscaped(_ index: String.Index, in text: String) -> Bool {
        var backslashCount = 0
        var cursor = index
        while cursor > text.startIndex {
            let previousIndex = text.index(before: cursor)
            guard text[previousIndex] == "\\" else { break }
            backslashCount += 1
            cursor = previousIndex
        }
        return backslashCount.isMultiple(of: 2) == false
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