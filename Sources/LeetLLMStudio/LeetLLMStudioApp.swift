import Foundation
import LeetLessonKit
import SwiftUI
import Textual
#if canImport(Darwin)
import Darwin
#endif

@main
struct LeetLLMStudioApp: App {
    @AppStorage(StudioTextSize.storageKey) private var textSize = StudioTextSize.defaultValue
    private let isRunningDiagramSmokeTest = CommandLine.arguments.contains(
        "--diagram-smoke-test"
    )

    var body: some Scene {
        WindowGroup("LeetLLM Studio") {
            if isRunningDiagramSmokeTest {
                DiagramSmokeTestView()
                    .frame(width: 900, height: 560)
            } else {
                StudioRootView(textSize: $textSize)
                    .frame(minWidth: 960, minHeight: 640)
            }
        }
        .defaultSize(width: 1_280, height: 800)
        .commands {
            CommandGroup(after: .textEditing) {
                Divider()
                Button("Make Text Bigger") {
                    StudioTextSize.increase(&textSize)
                }
                .keyboardShortcut("+", modifiers: .command)
                .disabled(!StudioTextSize.canIncrease(textSize))

                Button("Make Text Smaller") {
                    StudioTextSize.decrease(&textSize)
                }
                .keyboardShortcut("-", modifiers: .command)
                .disabled(!StudioTextSize.canDecrease(textSize))

                Button("Actual Text Size") {
                    textSize = StudioTextSize.defaultValue
                }
                .keyboardShortcut("0", modifiers: .command)
                .disabled(textSize == StudioTextSize.defaultValue)
            }
        }
    }
}

private struct DiagramSmokeTestView: View {
    private let fixture: DiagramSmokeFixture?
    private let setupError: String?
    @State private var didFinish = false

    init() {
        do {
            fixture = try DiagramSmokeFixture.load()
            setupError = nil
        } catch {
            fixture = nil
            setupError = error.localizedDescription
        }
    }

    var body: some View {
        Group {
            if let fixture {
                MermaidDiagramView(
                    id: fixture.id,
                    source: fixture.source,
                    title: fixture.title,
                    verifySnapshot: true,
                    onRenderEvent: finish
                )
            } else {
                Text(setupError ?? "The packaged diagram fixture is unavailable.")
            }
        }
        .task {
            if let setupError {
                finish(.failed(setupError))
                return
            }
            try? await Task.sleep(for: .seconds(20))
            guard !Task.isCancelled, !didFinish else { return }
            finish(.failed("The signed diagram smoke test timed out after 20 seconds."))
        }
    }

    private func finish(_ event: DiagramRenderEvent) {
        guard !didFinish else { return }
        didFinish = true

        switch event {
        case let .rendered(metrics):
            let expectedLabels = ["Prompt text", "Tokenizer", "Select the next token"]
            let missingLabels = expectedLabels.filter { !metrics.text.contains($0) }
            guard missingLabels.isEmpty, let visiblePixelCount = metrics.visiblePixelCount else {
                Self.exit(
                    status: 1,
                    message: "DIAGRAM_SMOKE_FAIL missing labels: \(missingLabels.joined(separator: ", "))"
                )
            }
            Self.exit(
                status: 0,
                message: "DIAGRAM_SMOKE_PASS svg=\(metrics.svgCount) graphics=\(metrics.graphicsCount) width=\(Int(metrics.width)) height=\(Int(metrics.height)) visiblePixels=\(visiblePixelCount)"
            )
        case let .failed(message):
            Self.exit(status: 1, message: "DIAGRAM_SMOKE_FAIL \(message)")
        }
    }

    private static func exit(status: Int32, message: String) -> Never {
        FileHandle.standardOutput.write(Data("\(message)\n".utf8))
        let resultURL = FileManager.default.temporaryDirectory
            .appending(path: "leetllm-diagram-smoke-result.txt")
        try? Data("\(message)\n".utf8).write(to: resultURL, options: .atomic)
        Darwin.exit(status)
    }
}

private struct DiagramSmokeFixture {
    let id: String
    let source: String
    let title: String

    static func load() throws -> Self {
        guard let courseRoot = Bundle.main.resourceURL?
            .appending(path: "Course", directoryHint: .isDirectory)
        else { throw DiagramSmokeError.missingCourse }

        let lesson = try LessonCatalog.discover(in: courseRoot).first { $0.id == "000" }
        guard let lesson else { throw DiagramSmokeError.missingLesson }
        guard let diagram = lesson.activities.first(where: {
            $0.id == "p000-token-generation-loop" && $0.kind == "mermaid"
        }) else { throw DiagramSmokeError.missingDiagram }

        return Self(id: diagram.id, source: diagram.configuration, title: "\(lesson.title) diagram")
    }
}

private enum DiagramSmokeError: Error, LocalizedError {
    case missingCourse
    case missingLesson
    case missingDiagram

    var errorDescription: String? {
        switch self {
        case .missingCourse:
            "The packaged Course resource is missing."
        case .missingLesson:
            "The packaged Start Here lesson is missing."
        case .missingDiagram:
            "The packaged Start Here token-generation diagram is missing."
        }
    }
}

enum StudioTextSize {
    static let storageKey = "studio.textSize"
    static let defaultValue = 1.0
    static let minimum = 0.8
    static let maximum = 2.0
    static let step = 0.1

    static func canIncrease(_ value: Double) -> Bool {
        value < maximum
    }

    static func canDecrease(_ value: Double) -> Bool {
        value > minimum
    }

    static func increase(_ value: inout Double) {
        value = min(maximum, rounded(value + step))
    }

    static func decrease(_ value: inout Double) {
        value = max(minimum, rounded(value - step))
    }

    static func dynamicTypeSize(for value: Double) -> DynamicTypeSize {
        switch value {
        case ..<0.9: .small
        case ..<1.0: .medium
        case ..<1.1: .large
        case ..<1.2: .xLarge
        case ..<1.3: .xxLarge
        case ..<1.4: .xxxLarge
        case ..<1.5: .accessibility1
        case ..<1.6: .accessibility2
        case ..<1.8: .accessibility3
        case ..<2.0: .accessibility4
        default: .accessibility5
        }
    }

    private static func rounded(_ value: Double) -> Double {
        (value * 10).rounded() / 10
    }
}

private struct StudioRootView: View {
    @Binding var textSize: Double
    @StateObject private var workspaceAuthorization = WorkspaceAuthorizationController()
    @State private var catalog = LessonCatalogSnapshot(
        lessons: [],
        diagnostics: [],
        revisionHash: ""
    )
    @State private var selectedLessonURL: URL?
    @State private var searchText = ""
    @State private var loadError: String?
    @State private var isShowingDiagnostics = false

    var body: some View {
        NavigationSplitView {
            VStack(spacing: 0) {
                List(filteredLessons, id: \.sourceURL, selection: $selectedLessonURL) { lesson in
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 6) {
                            Text(lesson.id)
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                            if !lesson.diagnostics.isEmpty {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                                    .accessibilityLabel("Has author diagnostics")
                            }
                        }
                        Text(lesson.title)
                            .font(.body.weight(.medium))
                            .lineLimit(2)
                    }
                    .padding(.vertical, 3)
                    .tag(lesson.sourceURL)
                }
                .overlay {
                    if !searchText.isEmpty && filteredLessons.isEmpty {
                        ContentUnavailableView.search(text: searchText)
                    }
                }

                Divider()

                HStack {
                    Text("\(catalog.lessons.count) lessons")
                        .foregroundStyle(.secondary)
                    Spacer()
                    if diagnosticCount > 0 {
                        Button {
                            isShowingDiagnostics = true
                        } label: {
                            Label("\(diagnosticCount)", systemImage: "exclamationmark.triangle")
                        }
                        .buttonStyle(.borderless)
                        .help("Show author diagnostics")
                    }
                }
                .font(.caption)
                .padding(.horizontal, 12)
                .frame(height: 34)
            }
            .navigationTitle("LeetLLM")
            .searchable(text: $searchText, prompt: "Search lessons")
            .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 280)
        } detail: {
            if let lesson = selectedLesson {
                LessonWorkspaceView(lesson: lesson, textSize: textSize)
            } else if let loadError {
                ContentUnavailableView(
                    "Lessons unavailable",
                    systemImage: "exclamationmark.triangle",
                    description: Text(loadError)
                )
            } else {
                ContentUnavailableView(
                    "Choose a lesson",
                    systemImage: "text.book.closed",
                    description: Text("Select a lesson from the curriculum.")
                )
            }
        }
        .task { await observeLessons() }
        .sheet(isPresented: $isShowingDiagnostics) {
            CatalogDiagnosticsView(catalog: catalog)
        }
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Menu {
                    if let authorization = workspaceAuthorization.authorization {
                        Text(authorization.rootURL.path)
                        Divider()
                        Button("Choose Different Folder…") {
                            workspaceAuthorization.chooseFolder()
                        }
                        Button("Forget Build Folder", role: .destructive) {
                            workspaceAuthorization.forgetFolder()
                        }
                    } else {
                        Button("Choose Build Folder…") {
                            workspaceAuthorization.chooseFolder()
                        }
                    }
                } label: {
                    Label(
                        workspaceAuthorization.selectedFolderName ?? "Build Folder",
                        systemImage: "folder.badge.gearshape"
                    )
                }
                .help("Choose where learner code may be written and executed")
            }

            ToolbarItem(placement: .automatic) {
                Menu {
                    Button("Make Text Bigger") {
                        StudioTextSize.increase(&textSize)
                    }
                    .disabled(!StudioTextSize.canIncrease(textSize))

                    Button("Make Text Smaller") {
                        StudioTextSize.decrease(&textSize)
                    }
                    .disabled(!StudioTextSize.canDecrease(textSize))

                    Divider()

                    Button("Actual Size") {
                        textSize = StudioTextSize.defaultValue
                    }
                    .disabled(textSize == StudioTextSize.defaultValue)
                } label: {
                    Label(
                        "Text Size \(Int((textSize * 100).rounded())) percent",
                        systemImage: "textformat.size"
                    )
                }
                .help("Change lesson and editor text size")
                .accessibilityLabel("Text size")
                .accessibilityValue("\(Int((textSize * 100).rounded())) percent")
            }
        }
        .environment(\.dynamicTypeSize, StudioTextSize.dynamicTypeSize(for: textSize))
        .environmentObject(workspaceAuthorization)
    }

    private var selectedLesson: LessonDocument? {
        catalog.lessons.first { $0.sourceURL == selectedLessonURL }
    }

    private var filteredLessons: [LessonDocument] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return catalog.lessons }
        return catalog.lessons.filter { lesson in
            [
                lesson.id,
                lesson.title,
                lesson.summary ?? "",
                lesson.module ?? "",
                lesson.tags.joined(separator: " "),
                lesson.markdown,
            ].contains { $0.localizedCaseInsensitiveContains(query) }
        }
    }

    private var diagnosticCount: Int {
        catalog.diagnostics.count + catalog.lessons.reduce(0) { $0 + $1.diagnostics.count }
    }

    private var contentRoots: [LessonContentRoot] {
        let environment = ProcessInfo.processInfo.environment
        let configuredPaths = environment["LEETLLM_COURSE_ROOTS"]?
            .split(separator: ":")
            .map(String.init)
        let paths = configuredPaths?.isEmpty == false
            ? configuredPaths!
            : [
                environment["LEETLLM_COURSE_ROOT"]
                    ?? bundledCourseRoot?.path
                    ?? FileManager.default.currentDirectoryPath,
            ]
        return paths.enumerated().map { index, path in
            let url = URL(fileURLWithPath: path, isDirectory: true)
            return LessonContentRoot(id: "root-\(index + 1)", url: url)
        }
    }

    private var bundledCourseRoot: URL? {
        guard let courseRoot = Bundle.main.resourceURL?
            .appending(path: "Course", directoryHint: .isDirectory),
            FileManager.default.fileExists(
                atPath: courseRoot.appending(path: "Problems", directoryHint: .isDirectory).path
            )
        else { return nil }
        return courseRoot
    }

    @MainActor
    private func observeLessons() async {
        do {
            let roots = contentRoots
            let initialCatalog = try await Task.detached {
                try LessonCatalog.load(from: roots)
            }.value
            apply(initialCatalog)

            for await updatedCatalog in LessonCatalog.updates(from: roots) {
                apply(updatedCatalog)
            }
        } catch {
            loadError = error.localizedDescription
        }
    }

    @MainActor
    private func apply(_ updatedCatalog: LessonCatalogSnapshot) {
        guard updatedCatalog.revisionHash != catalog.revisionHash else { return }
        catalog = updatedCatalog
        if let selectedLessonURL,
            updatedCatalog.lessons.contains(where: { $0.sourceURL == selectedLessonURL })
        {
            return
        }
        selectedLessonURL = updatedCatalog.lessons.first?.sourceURL
        loadError = updatedCatalog.lessons.isEmpty
            ? "No Markdown lessons were found in the configured content roots."
            : nil
    }
}

struct LessonReader: View {
    let lesson: LessonDocument
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var previewedDocument: PreviewedLocalDocument?

    var body: some View {
        ScrollViewReader { proxy in
            GeometryReader { geometry in
                HStack(spacing: 0) {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 18) {
                            HStack(spacing: 8) {
                                Text("Problem \(lesson.id)")
                                if !lesson.activities.isEmpty {
                                    Text("\(lesson.activities.count) activities")
                                }
                            }
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                            if !lesson.diagnostics.isEmpty {
                                LessonDiagnosticBanner(diagnostics: lesson.diagnostics)
                            }

                            ForEach(
                                Array(LessonMarkdownRendering.blocks(in: lesson).enumerated()),
                                id: \.offset
                            ) { _, block in
                                switch block {
                                case let .markdown(markdown):
                                    StructuredText(
                                        markdown: markdown,
                                        baseURL: lesson.sourceURL.deletingLastPathComponent(),
                                        syntaxExtensions: [.math]
                                    )
                                    .textual.structuredTextStyle(.gitHub)
                                    .textual.textSelection(.enabled)
                                    .textual.imageAttachmentLoader(
                                        .image(
                                            relativeTo: lesson.sourceURL.deletingLastPathComponent()
                                        )
                                    )
                                case let .mermaid(id, source):
                                    MermaidDiagramView(
                                        id: id,
                                        source: source,
                                        title: "\(lesson.title) diagram"
                                    )
                                case let .checklist(anchor, title, items):
                                    LessonChecklistView(
                                        lessonID: lesson.id,
                                        contentVersion: lesson.contentVersion,
                                        anchor: anchor,
                                        title: title,
                                        items: items
                                    )
                                }
                            }
                        }
                        .frame(maxWidth: 860, alignment: .leading)
                        .padding(.horizontal, geometry.size.width < 700 ? 20 : 32)
                        .padding(.vertical, 28)
                        .frame(maxWidth: .infinity, alignment: .top)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                    if lesson.sections.count > 1, geometry.size.width >= 1_000 {
                        Divider()
                        LessonOutline(sections: lesson.sections) { section in
                            scroll(to: section, with: proxy)
                        }
                        .frame(width: 210)
                        .background(.background.secondary)
                    }
                }
                .toolbar {
                    if lesson.sections.count > 1, geometry.size.width < 1_000 {
                        ToolbarItem(placement: .primaryAction) {
                            Menu {
                                ForEach(lesson.sections) { section in
                                    Button(section.title) {
                                        scroll(to: section, with: proxy)
                                    }
                                }
                            } label: {
                                Label("Sections", systemImage: "list.bullet")
                            }
                            .help("Navigate lesson sections")
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle(lesson.title)
        .environment(\.openURL, OpenURLAction { url in
            if let document = PreviewedLocalDocument(resolving: url) {
                previewedDocument = document
                return .handled
            }
            return .systemAction
        })
        .sheet(item: $previewedDocument) { document in
            LocalDocumentPreviewSheet(document: document)
        }
    }

    private func scroll(to section: LessonSection, with proxy: ScrollViewProxy) {
        if reduceMotion {
            proxy.scrollTo(section.anchor, anchor: .top)
        } else {
            withAnimation(.easeInOut(duration: 0.2)) {
                proxy.scrollTo(section.anchor, anchor: .top)
            }
        }
    }
}

struct PreviewedLocalDocument: Identifiable {
    enum Kind: Equatable {
        case markdown
        case source(language: String)
    }

    let id: String
    let title: String
    let contents: String
    let baseURL: URL
    let kind: Kind

    init?(resolving url: URL) {
        let absolute = url.absoluteURL
        guard absolute.isFileURL,
              let kind = Self.kind(forExtension: absolute.pathExtension)
        else {
            return nil
        }
        let fileURL = URL(fileURLWithPath: absolute.path).standardizedFileURL
        guard let contents = try? String(contentsOf: fileURL, encoding: .utf8) else {
            return nil
        }
        self.id = fileURL.path
        self.contents = contents
        self.baseURL = fileURL.deletingLastPathComponent()
        self.kind = kind
        switch kind {
        case .markdown:
            self.title = Self.firstHeading(in: contents)
                ?? fileURL.deletingPathExtension().lastPathComponent
        case .source:
            self.title = fileURL.lastPathComponent
        }
    }

    private static func kind(forExtension pathExtension: String) -> Kind? {
        switch pathExtension.lowercased() {
        case "md":
            .markdown
        case "swift":
            .source(language: "swift")
        case "metal":
            .source(language: "metal")
        default:
            nil
        }
    }

    private static func firstHeading(in markdown: String) -> String? {
        for line in markdown.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("#") else { continue }
            let heading = trimmed.drop { $0 == "#" }.trimmingCharacters(in: .whitespaces)
            if !heading.isEmpty { return heading }
        }
        return nil
    }
}

private struct LocalDocumentPreviewSheet: View {
    let document: PreviewedLocalDocument
    @Environment(\.dismiss) private var dismiss
    @AppStorage(StudioTextSize.storageKey) private var textSize = StudioTextSize.defaultValue

    var body: some View {
        NavigationStack {
            Group {
                switch document.kind {
                case .markdown:
                    ScrollView {
                        StructuredText(
                            markdown: document.contents,
                            baseURL: document.baseURL,
                            syntaxExtensions: [.math]
                        )
                        .textual.structuredTextStyle(.gitHub)
                        .textual.textSelection(.enabled)
                        .textual.imageAttachmentLoader(.image(relativeTo: document.baseURL))
                        .frame(maxWidth: 860, alignment: .leading)
                        .padding(.horizontal, 32)
                        .padding(.vertical, 28)
                        .frame(maxWidth: .infinity, alignment: .top)
                    }
                case let .source(language):
                    CodeEditorView(
                        text: .constant(document.contents),
                        documentID: document.id,
                        language: language,
                        textScale: textSize,
                        isEditable: false
                    )
                }
            }
            .frame(minWidth: 720, minHeight: 560)
            .navigationTitle(document.title)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private struct LessonChecklistView: View {
    let anchor: String
    let title: String
    let items: [LessonChecklistItem]

    @State private var completedItemIDs: Set<String>
    private let persistenceKey: String

    init(
        lessonID: String,
        contentVersion: Int,
        anchor: String,
        title: String,
        items: [LessonChecklistItem]
    ) {
        self.anchor = anchor
        self.title = title
        self.items = items
        self.persistenceKey = "studio.checklist.\(lessonID).v\(contentVersion).\(anchor)"

        let storedIDs = UserDefaults.standard.stringArray(forKey: persistenceKey)
        let authoredIDs = items.filter(\.isCompleted).map(\.id)
        let validItemIDs = Set(items.map(\.id))
        _completedItemIDs = State(
            initialValue: Set(storedIDs ?? authoredIDs).intersection(validItemIDs)
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                    .font(.title2.weight(.semibold))
                Spacer()
                Text("\(completedItemIDs.count) of \(items.count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Divider()

            ProgressView(value: Double(completedItemIDs.count), total: Double(items.count))
                .accessibilityLabel("Checklist progress")
                .accessibilityValue("\(completedItemIDs.count) of \(items.count) complete")

            VStack(alignment: .leading, spacing: 10) {
                ForEach(items) { item in
                    Toggle(isOn: completionBinding(for: item.id)) {
                        checklistLabel(item.text)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .toggleStyle(.checkbox)
                }
            }
        }
        .id(anchor)
        .accessibilityElement(children: .contain)
    }

    private func completionBinding(for itemID: String) -> Binding<Bool> {
        Binding(
            get: { completedItemIDs.contains(itemID) },
            set: { isCompleted in
                if isCompleted {
                    completedItemIDs.insert(itemID)
                } else {
                    completedItemIDs.remove(itemID)
                }
                UserDefaults.standard.set(
                    completedItemIDs.sorted(),
                    forKey: persistenceKey
                )
            }
        )
    }

    @ViewBuilder
    private func checklistLabel(_ markdown: String) -> some View {
        if let attributed = try? AttributedString(
            markdown: markdown,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            Text(attributed)
        } else {
            Text(markdown)
        }
    }
}

private struct LessonOutline: View {
    let sections: [LessonSection]
    let select: (LessonSection) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                Text("ON THIS PAGE")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 6)

                ForEach(sections) { section in
                    Button {
                        select(section)
                    } label: {
                        Text(section.title)
                            .font(.caption)
                            .foregroundStyle(.primary)
                            .multilineTextAlignment(.leading)
                            .lineLimit(3)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.leading, CGFloat(max(0, section.level - 1)) * 10)
                            .padding(.vertical, 3)
                    }
                    .buttonStyle(.plain)
                    .help("Go to line \(section.sourceLine)")
                }
            }
            .padding(16)
        }
    }
}

private struct LessonDiagnosticBanner: View {
    let diagnostics: [LessonDiagnostic]

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(diagnostics.enumerated()), id: \.offset) { _, diagnostic in
                    Text(diagnostic.line.map { "Line \($0): \(diagnostic.message)" }
                        ?? diagnostic.message)
                }
            }
            .font(.caption)
            .padding(.top, 8)
        } label: {
            Label(
                "\(diagnostics.count) author diagnostic\(diagnostics.count == 1 ? "" : "s")",
                systemImage: "exclamationmark.triangle.fill"
            )
            .font(.subheadline.weight(.semibold))
        }
        .tint(.orange)
        .padding(12)
        .background(.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

private struct CatalogDiagnosticsView: View {
    let catalog: LessonCatalogSnapshot
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if !catalog.diagnostics.isEmpty {
                    Section("Catalog") {
                        ForEach(catalog.diagnostics) { diagnostic in
                            DiagnosticRow(
                                severity: diagnostic.severity,
                                message: diagnostic.message,
                                detail: diagnostic.sourceURLs.map(\.path).joined(separator: "\n")
                            )
                        }
                    }
                }

                ForEach(catalog.lessons.filter { !$0.diagnostics.isEmpty }, id: \.sourceURL) { lesson in
                    Section(lesson.title) {
                        ForEach(Array(lesson.diagnostics.enumerated()), id: \.offset) { _, diagnostic in
                            DiagnosticRow(
                                severity: diagnostic.severity,
                                message: diagnostic.message,
                                detail: diagnostic.line.map { "\(lesson.relativePath):\($0)" }
                                    ?? lesson.relativePath
                            )
                        }
                    }
                }
            }
            .navigationTitle("Author Diagnostics")
            .frame(minWidth: 620, minHeight: 420)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private struct DiagnosticRow: View {
    let severity: LessonDiagnosticSeverity
    let message: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: severity == .error
                ? "xmark.octagon.fill"
                : "exclamationmark.triangle.fill")
                .foregroundStyle(severity == .error ? .red : .orange)
                .accessibilityLabel(severity.rawValue.capitalized)
            VStack(alignment: .leading, spacing: 3) {
                Text(message)
                if !detail.isEmpty {
                    Text(detail)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
    }
}