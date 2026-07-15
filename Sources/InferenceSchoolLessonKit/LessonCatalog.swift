import CryptoKit
import Foundation
import Yams

public enum LessonDiagnosticSeverity: String, Hashable, Sendable {
    case warning
    case error
}

public struct LessonDiagnostic: Hashable, Sendable {
    public let code: String
    public let severity: LessonDiagnosticSeverity
    public let message: String
    public let line: Int?

    public init(
        code: String = "lesson",
        severity: LessonDiagnosticSeverity,
        message: String,
        line: Int? = nil
    ) {
        self.code = code
        self.severity = severity
        self.message = message
        self.line = line
    }
}

public struct LessonSection: Identifiable, Hashable, Sendable {
    public let id: String
    public let anchor: String
    public let title: String
    public let level: Int
    public let sourceLine: Int

    public init(id: String, anchor: String, title: String, level: Int, sourceLine: Int) {
        self.id = id
        self.anchor = anchor
        self.title = title
        self.level = level
        self.sourceLine = sourceLine
    }
}

public struct LessonActivity: Identifiable, Hashable, Sendable {
    public let id: String
    public let kind: String
    public let schemaVersion: Int
    public let sourceLines: ClosedRange<Int>
    public let configuration: String

    public init(
        id: String,
        kind: String,
        schemaVersion: Int,
        sourceLines: ClosedRange<Int>,
        configuration: String
    ) {
        self.id = id
        self.kind = kind
        self.schemaVersion = schemaVersion
        self.sourceLines = sourceLines
        self.configuration = configuration
    }
}

public struct LessonDocument: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let order: Int?
    public let module: String?
    public let summary: String?
    public let prerequisites: [String]
    public let tags: [String]
    public let capabilities: [String]
    public let estimatedMinutes: Int?
    public let contentVersion: Int
    public let contentRootID: String?
    public let sourceURL: URL
    public let relativePath: String
    public let markdown: String
    public let sections: [LessonSection]
    public let activities: [LessonActivity]
    public let diagnostics: [LessonDiagnostic]

    public init(
        id: String,
        title: String,
        order: Int?,
        module: String? = nil,
        summary: String? = nil,
        prerequisites: [String] = [],
        tags: [String] = [],
        capabilities: [String] = [],
        estimatedMinutes: Int? = nil,
        contentVersion: Int = 1,
        contentRootID: String? = nil,
        sourceURL: URL,
        relativePath: String,
        markdown: String,
        sections: [LessonSection] = [],
        activities: [LessonActivity] = [],
        diagnostics: [LessonDiagnostic] = []
    ) {
        self.id = id
        self.title = title
        self.order = order
        self.module = module
        self.summary = summary
        self.prerequisites = prerequisites
        self.tags = tags
        self.capabilities = capabilities
        self.estimatedMinutes = estimatedMinutes
        self.contentVersion = contentVersion
        self.contentRootID = contentRootID
        self.sourceURL = sourceURL
        self.relativePath = relativePath
        self.markdown = markdown
        self.sections = sections
        self.activities = activities
        self.diagnostics = diagnostics
    }
}

public struct LessonContentRoot: Identifiable, Hashable, Sendable {
    public let id: String
    public let url: URL

    public init(id: String, url: URL) {
        self.id = id
        self.url = url
    }
}

public struct LessonCatalogDiagnostic: Identifiable, Hashable, Sendable {
    public let code: String
    public let severity: LessonDiagnosticSeverity
    public let message: String
    public let lessonID: String?
    public let sourceURLs: [URL]

    public var id: String {
        ([code, lessonID ?? ""] + sourceURLs.map(\.path)).joined(separator: ":")
    }

    public init(
        code: String,
        severity: LessonDiagnosticSeverity,
        message: String,
        lessonID: String? = nil,
        sourceURLs: [URL] = []
    ) {
        self.code = code
        self.severity = severity
        self.message = message
        self.lessonID = lessonID
        self.sourceURLs = sourceURLs
    }
}

public struct LessonCatalogSnapshot: Hashable, Sendable {
    public let lessons: [LessonDocument]
    public let diagnostics: [LessonCatalogDiagnostic]
    public let revisionHash: String

    public init(
        lessons: [LessonDocument],
        diagnostics: [LessonCatalogDiagnostic],
        revisionHash: String
    ) {
        self.lessons = lessons
        self.diagnostics = diagnostics
        self.revisionHash = revisionHash
    }
}

public enum LessonCatalog {
    private struct FrontMatter: Decodable {
        let formatVersion: Int?
        let id: String?
        let title: String?
        let order: Int?
        let module: String?
        let summary: String?
        let prerequisites: [String]?
        let tags: [String]?
        let capabilities: [String]?
        let estimatedMinutes: Int?
        let contentVersion: Int?
    }

    private struct ParsedMarkdown {
        let frontMatter: FrontMatter?
        let body: String
        let diagnostics: [LessonDiagnostic]
    }

    private struct ParsedActivities {
        let activities: [LessonActivity]
        let diagnostics: [LessonDiagnostic]
    }

    private struct ScanResult {
        let lessons: [LessonDocument]
        let diagnostics: [LessonCatalogDiagnostic]
        let revisionMaterial: [String]
    }

    private static let activityKinds: Set<String> = [
        "quiz", "numeric", "ordering", "prediction", "swift", "metal",
        "exercise", "tests",
        "tensor", "matrix", "attention", "kv-cache", "quantization",
        "benchmark", "roofline", "trace", "inference", "mermaid",
    ]

    public static func discover(in contentRoot: URL) throws -> [LessonDocument] {
        sorted(try scan(contentRoot, rootID: nil).lessons)
    }

    public static func load(from contentRoots: [LessonContentRoot]) throws -> LessonCatalogSnapshot {
        var lessons: [LessonDocument] = []
        var diagnostics: [LessonCatalogDiagnostic] = []
        var revisionMaterial: [String] = []

        for contentRoot in contentRoots {
            let result = try scan(contentRoot.url, rootID: contentRoot.id)
            lessons.append(contentsOf: result.lessons)
            diagnostics.append(contentsOf: result.diagnostics)
            revisionMaterial.append(contentsOf: result.revisionMaterial)
        }

        for (lessonID, duplicates) in Dictionary(grouping: lessons, by: \.id)
            where duplicates.count > 1
        {
            let sourceURLs = duplicates.map(\.sourceURL).sorted { $0.path < $1.path }
            diagnostics.append(LessonCatalogDiagnostic(
                code: "duplicate-lesson-id",
                severity: .error,
                message: "Lesson ID '\(lessonID)' is declared by \(duplicates.count) files.",
                lessonID: lessonID,
                sourceURLs: sourceURLs
            ))
        }

        return LessonCatalogSnapshot(
            lessons: sorted(lessons),
            diagnostics: diagnostics.sorted { $0.id < $1.id },
            revisionHash: revisionHash(for: revisionMaterial)
        )
    }

    public static func updates(
        from contentRoots: [LessonContentRoot],
        every interval: Duration = .seconds(1)
    ) -> AsyncStream<LessonCatalogSnapshot> {
        AsyncStream { continuation in
            let task = Task.detached {
                var previousRevision: String?
                while !Task.isCancelled {
                    if let snapshot = try? load(from: contentRoots),
                        snapshot.revisionHash != previousRevision
                    {
                        previousRevision = snapshot.revisionHash
                        continuation.yield(snapshot)
                    }
                    do {
                        try await Task.sleep(for: interval)
                    } catch {
                        break
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private static func scan(_ contentRoot: URL, rootID: String?) throws -> ScanResult {
        let fileManager = FileManager.default
        let contentRoot = contentRoot.standardizedFileURL.resolvingSymlinksInPath()
        let lessonsRoot = fileManager.fileExists(
            atPath: contentRoot.appending(path: "Problems").path
        ) ? contentRoot.appending(path: "Problems") : contentRoot

        guard let enumerator = fileManager.enumerator(
            at: lessonsRoot,
            includingPropertiesForKeys: [.isRegularFileKey, .isHiddenKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return ScanResult(
                lessons: [],
                diagnostics: [LessonCatalogDiagnostic(
                    code: "unreadable-content-root",
                    severity: .error,
                    message: "The content root could not be read.",
                    sourceURLs: [contentRoot]
                )],
                revisionMaterial: [contentRoot.path]
            )
        }

        var lessons: [LessonDocument] = []
        var revisionMaterial: [String] = []
        for case let sourceURL as URL in enumerator {
            guard sourceURL.pathExtension.lowercased() == "md" else { continue }
            let values = try sourceURL.resourceValues(forKeys: [.isRegularFileKey, .isHiddenKey])
            guard values.isRegularFile == true, values.isHidden != true else { continue }

            let source = try String(contentsOf: sourceURL, encoding: .utf8)
            revisionMaterial.append(sourceURL.path + "\0" + source)
            let parsed = parseMarkdown(source)
            let parsedActivities = activities(in: parsed.body)
            let relativePath = relativePath(from: lessonsRoot, to: sourceURL)
            let pathComponent = sourceURL.deletingLastPathComponent().lastPathComponent
            let inferredID = leadingNumber(in: pathComponent)
                ?? [rootID, relativePath.replacingOccurrences(of: "/", with: ".")]
                    .compactMap { $0 }
                    .joined(separator: ".")
            let lessonID = parsed.frontMatter?.id ?? inferredID
            let title = parsed.frontMatter?.title
                ?? firstHeading(in: parsed.body)
                ?? sourceURL.deletingPathExtension().lastPathComponent

            lessons.append(LessonDocument(
                id: lessonID,
                title: normalizedTitle(title),
                order: parsed.frontMatter?.order ?? Int(inferredID),
                module: parsed.frontMatter?.module,
                summary: parsed.frontMatter?.summary ?? firstParagraph(in: parsed.body),
                prerequisites: parsed.frontMatter?.prerequisites ?? [],
                tags: parsed.frontMatter?.tags ?? [],
                capabilities: parsed.frontMatter?.capabilities ?? [],
                estimatedMinutes: parsed.frontMatter?.estimatedMinutes,
                contentVersion: parsed.frontMatter?.contentVersion ?? 1,
                contentRootID: rootID,
                sourceURL: sourceURL,
                relativePath: relativePath,
                markdown: parsed.body,
                sections: sections(in: parsed.body),
                activities: parsedActivities.activities,
                diagnostics: parsed.diagnostics + parsedActivities.diagnostics
            ))
        }

        return ScanResult(lessons: lessons, diagnostics: [], revisionMaterial: revisionMaterial)
    }

    private static func sorted(_ lessons: [LessonDocument]) -> [LessonDocument] {
        lessons.sorted { left, right in
            switch (left.order, right.order) {
            case let (.some(leftOrder), .some(rightOrder)) where leftOrder != rightOrder:
                return leftOrder < rightOrder
            case (.some, .none):
                return true
            case (.none, .some):
                return false
            default:
                let titleOrder = left.title.localizedStandardCompare(right.title)
                return titleOrder == .orderedSame
                    ? left.sourceURL.path < right.sourceURL.path
                    : titleOrder == .orderedAscending
            }
        }
    }

    private static func revisionHash(for material: [String]) -> String {
        var hasher = SHA256()
        for value in material.sorted() {
            hasher.update(data: Data(value.utf8))
            hasher.update(data: Data([0]))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func relativePath(from root: URL, to source: URL) -> String {
        let rootComponents = root.standardizedFileURL.resolvingSymlinksInPath().pathComponents
        let sourceComponents = source.standardizedFileURL.resolvingSymlinksInPath().pathComponents
        guard sourceComponents.starts(with: rootComponents) else {
            return source.lastPathComponent
        }
        return sourceComponents.dropFirst(rootComponents.count).joined(separator: "/")
    }

    private static func parseMarkdown(_ source: String) -> ParsedMarkdown {
        let lines = source.split(separator: "\n", omittingEmptySubsequences: false)
        guard lines.first == "---" else {
            return ParsedMarkdown(frontMatter: nil, body: source, diagnostics: [])
        }
        guard let closingIndex = lines.dropFirst().firstIndex(where: { $0 == "---" }) else {
            return ParsedMarkdown(
                frontMatter: nil,
                body: source,
                diagnostics: [LessonDiagnostic(
                    severity: .error,
                    message: "Front matter starts with --- but has no closing delimiter."
                )]
            )
        }

        let yaml = lines[1..<closingIndex].joined(separator: "\n")
        let bodyStart = lines.index(after: closingIndex)
        let body = lines[bodyStart...].joined(separator: "\n")
        do {
            let metadata = try YAMLDecoder().decode(FrontMatter.self, from: yaml)
            let diagnostics = metadata.formatVersion.map { version in
                version == 1 ? [] : [LessonDiagnostic(
                    severity: .warning,
                    message: "Front-matter format version \(version) is not fully supported."
                )]
            } ?? []
            return ParsedMarkdown(frontMatter: metadata, body: body, diagnostics: diagnostics)
        } catch {
            return ParsedMarkdown(
                frontMatter: nil,
                body: body,
                diagnostics: [LessonDiagnostic(
                    severity: .error,
                    message: "Invalid YAML front matter: \(error.localizedDescription)"
                )]
            )
        }
    }

    private static func activities(in markdown: String) -> ParsedActivities {
        let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var activities: [LessonActivity] = []
        var diagnostics: [LessonDiagnostic] = []
        var activityIDs: Set<String> = []
        var lineIndex = 0

        while lineIndex < lines.count {
            let line = lines[lineIndex].trimmingCharacters(in: .whitespaces)
            guard line.hasPrefix("```") else {
                lineIndex += 1
                continue
            }

            let info = String(line.dropFirst(3))
            let declaredKind = info.split(whereSeparator: { $0.isWhitespace || $0 == "{" }).first.map(String.init) ?? ""
            let start = lineIndex
            var end = start + 1
            while end < lines.count && !lines[end].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                end += 1
            }
            let configuration = end > start + 1
                ? lines[(start + 1)..<min(end, lines.count)].joined(separator: "\n")
                : ""
            let isEditableCode = ["swift", "metal"].contains(declaredKind)
                && info.contains(".exercise")
            let isKnownDirective = activityKinds.contains(declaredKind)
                && (!["swift", "metal"].contains(declaredKind) || isEditableCode)
            let looksInteractive = info.contains(".exercise")
                || configuration.split(separator: "\n", maxSplits: 1).contains {
                    $0.trimmingCharacters(in: .whitespaces).hasPrefix("schemaVersion:")
                }

            guard isKnownDirective else {
                if looksInteractive && !declaredKind.isEmpty {
                    diagnostics.append(LessonDiagnostic(
                        code: "unknown-directive",
                        severity: .warning,
                        message: "Unknown directive '\(declaredKind)' renders as a code block.",
                        line: start + 1
                    ))
                }
                lineIndex = min(end + 1, lines.count)
                continue
            }

            let explicitID = firstMatch(#"#([A-Za-z0-9._-]+)"#, in: info)
            let activityID = explicitID ?? "\(declaredKind)-\(start + 1)"
            let schemaVersion = Int(firstMatch(#"schemaVersion:\s*(\d+)"#, in: configuration) ?? "1") ?? 1
            if explicitID == nil {
                diagnostics.append(LessonDiagnostic(
                    code: "missing-activity-id",
                    severity: .warning,
                    message: "Directive '\(declaredKind)' needs an explicit ID for stable progress.",
                    line: start + 1
                ))
            }
            guard schemaVersion == 1 else {
                diagnostics.append(LessonDiagnostic(
                    code: "unsupported-directive-version",
                    severity: .warning,
                    message: "Directive '\(declaredKind)' schema version \(schemaVersion) renders as a code block.",
                    line: start + 1
                ))
                lineIndex = min(end + 1, lines.count)
                continue
            }
            guard activityIDs.insert(activityID).inserted else {
                diagnostics.append(LessonDiagnostic(
                    code: "duplicate-activity-id",
                    severity: .error,
                    message: "Activity ID '\(activityID)' is declared more than once.",
                    line: start + 1
                ))
                lineIndex = min(end + 1, lines.count)
                continue
            }
            activities.append(LessonActivity(
                id: activityID,
                kind: declaredKind,
                schemaVersion: schemaVersion,
                sourceLines: (start + 1)...(min(end, lines.count - 1) + 1),
                configuration: configuration
            ))
            lineIndex = min(end + 1, lines.count)
        }
        return ParsedActivities(activities: activities, diagnostics: diagnostics)
    }

    private static func sections(in markdown: String) -> [LessonSection] {
        let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false)
        var sections: [LessonSection] = []
        var identifiers: [String: Int] = [:]
        var insideFence = false

        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                insideFence.toggle()
                continue
            }
            guard !insideFence else { continue }
            let marker = trimmed.prefix(while: { $0 == "#" })
            guard (1...6).contains(marker.count),
                trimmed.dropFirst(marker.count).first == " "
            else { continue }

            let title = trimmed.dropFirst(marker.count + 1)
                .trimmingCharacters(in: CharacterSet(charactersIn: " #\t"))
            let anchor = headingAnchor(title)
            let baseID = anchor
            let occurrence = identifiers[baseID, default: 0]
            identifiers[baseID] = occurrence + 1
            sections.append(LessonSection(
                id: occurrence == 0 ? baseID : "\(baseID)-\(occurrence + 1)",
                anchor: anchor,
                title: title,
                level: marker.count,
                sourceLine: index + 1
            ))
        }
        return sections
    }

    private static func headingAnchor(_ value: String) -> String {
        let anchor = String(
            value.lowercased()
                .map { $0.isWhitespace ? "-" : $0 }
                .filter { $0.isLetter || $0.isNumber || $0 == "-" }
                .split(separator: "-", omittingEmptySubsequences: true)
                .joined(separator: "-")
        )
        return anchor.isEmpty ? "section" : anchor
    }

    private static func leadingNumber(in value: String) -> String? {
        let digits = value.prefix(while: \Character.isNumber)
        return digits.isEmpty ? nil : String(digits)
    }

    private static func firstHeading(in markdown: String) -> String? {
        markdown.split(separator: "\n", omittingEmptySubsequences: false)
            .first { $0.hasPrefix("# ") }
            .map { String($0.dropFirst(2)).trimmingCharacters(in: .whitespaces) }
    }

    private static func firstParagraph(in markdown: String) -> String? {
        var paragraph: [Substring] = []
        for line in markdown.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                if !paragraph.isEmpty { break }
                continue
            }
            if trimmed.hasPrefix("#") || trimmed.hasPrefix("|") || trimmed.hasPrefix("```") {
                continue
            }
            paragraph.append(Substring(trimmed))
        }
        return paragraph.isEmpty ? nil : paragraph.joined(separator: " ")
    }

    private static func firstMatch(_ pattern: String, in value: String) -> String? {
        guard let expression = try? NSRegularExpression(pattern: pattern),
            let match = expression.firstMatch(
                in: value,
                range: NSRange(value.startIndex..., in: value)
            ),
            match.numberOfRanges > 1,
            let range = Range(match.range(at: 1), in: value)
        else { return nil }
        return String(value[range])
    }

    private static func normalizedTitle(_ title: String) -> String {
        title.replacingOccurrences(
            of: #"^(?:Problem\s+)?\d{3}:\s*"#,
            with: "",
            options: .regularExpression
        )
    }
}