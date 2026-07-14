import CryptoKit
import LeetLessonKit
import LeetLLMRuntime
import LeetRunnerClient
import LeetRunnerProtocol
import LeetWorkspaceKit
import SwiftUI

struct LessonWorkspaceView: View {
    let lesson: LessonDocument
    let textSize: Double

    var body: some View {
        if let activity = RuntimeRegistry.activity(forLessonID: lesson.id) {
            LessonActivityWorkspace(
                lesson: lesson,
                activity: activity,
                textSize: textSize
            )
            .id(lesson.id)
        } else {
            LessonReader(lesson: lesson, textSize: textSize)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct LessonActivityWorkspace: View {
    @EnvironmentObject private var workspaceAuthorization: WorkspaceAuthorizationController
    let lesson: LessonDocument
    let textSize: Double
    @AppStorage(WorkbenchPanel.storageKey) private var isWorkbenchCollapsed = false
    @StateObject private var model: LessonWorkbenchModel

    init(lesson: LessonDocument, activity: RuntimeActivityDescriptor, textSize: Double) {
        self.lesson = lesson
        self.textSize = textSize
        _model = StateObject(wrappedValue: LessonWorkbenchModel(
            lesson: lesson,
            activity: activity
        ))
    }

    var body: some View {
        GeometryReader { geometry in
            let placement = WorkbenchPanel.placement(forWidth: geometry.size.width)
            if isWorkbenchCollapsed {
                collapsedLayout(placement: placement)
            } else if placement == .trailing {
                HSplitView {
                    LessonReader(lesson: lesson, textSize: textSize)
                        .frame(minWidth: 520, idealWidth: 620)
                    LessonWorkbench(
                        model: model,
                        textSize: textSize,
                        placement: placement,
                        onCollapse: { isWorkbenchCollapsed = true }
                    )
                    .frame(minWidth: 680, idealWidth: 760)
                }
            } else {
                VSplitView {
                    LessonReader(lesson: lesson, textSize: textSize)
                        .frame(minHeight: 320, idealHeight: 470)
                    LessonWorkbench(
                        model: model,
                        textSize: textSize,
                        placement: placement,
                        onCollapse: { isWorkbenchCollapsed = true }
                    )
                    .frame(minHeight: 260, idealHeight: 360)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: workspaceAuthorization.authorization?.id) {
            if let authorization = workspaceAuthorization.authorization {
                await model.prepare(authorization: authorization)
            } else {
                model.revokeWorkspaceAuthorization()
            }
        }
    }

    @ViewBuilder
    private func collapsedLayout(placement: WorkbenchPanel.Placement) -> some View {
        if placement == .trailing {
            HStack(spacing: 0) {
                LessonReader(lesson: lesson, textSize: textSize)
                Divider()
                WorkbenchRestoreBar(placement: placement) {
                    isWorkbenchCollapsed = false
                }
            }
        } else {
            VStack(spacing: 0) {
                LessonReader(lesson: lesson, textSize: textSize)
                Divider()
                WorkbenchRestoreBar(placement: placement) {
                    isWorkbenchCollapsed = false
                }
            }
        }
    }
}

enum WorkbenchPanel {
    static let storageKey = "studio.workbench.isCollapsed"
    static let trailingLayoutMinimumWidth: CGFloat = 1_280

    enum Placement: Equatable {
        case trailing
        case bottom

        var collapseSymbol: String {
            self == .trailing ? "chevron.right" : "chevron.down"
        }

        var restoreSymbol: String {
            self == .trailing ? "chevron.left" : "chevron.up"
        }
    }

    static func placement(forWidth width: CGFloat) -> Placement {
        width >= trailingLayoutMinimumWidth ? .trailing : .bottom
    }
}

private struct WorkbenchRestoreBar: View {
    let placement: WorkbenchPanel.Placement
    let restore: () -> Void

    var body: some View {
        Group {
            if placement == .trailing {
                VStack {
                    restoreButton
                    Spacer()
                }
                .padding(.vertical, 8)
                .frame(width: 42)
            } else {
                HStack {
                    Spacer()
                    restoreButton
                    Spacer()
                }
                .frame(height: 42)
            }
        }
        .background(.background)
    }

    private var restoreButton: some View {
        Button(action: restore) {
            if placement == .bottom {
                Label("Show workbench", systemImage: placement.restoreSymbol)
                    .labelStyle(.titleAndIcon)
            } else {
                Label("Show workbench", systemImage: placement.restoreSymbol)
                    .labelStyle(.iconOnly)
            }
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
        .accessibilityLabel("Show workbench")
        .help("Show the code and build panel")
    }
}

private struct LessonWorkbench: View {
    @EnvironmentObject private var workspaceAuthorization: WorkspaceAuthorizationController
    @ObservedObject var model: LessonWorkbenchModel
    @State private var isConfirmingReset = false
    let textSize: Double
    let placement: WorkbenchPanel.Placement
    let onCollapse: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            workbenchToolbar
            Divider()
            workbenchContent
        }
        .background(.background)
        .confirmationDialog(
            "Reset this file to the course starter?",
            isPresented: $isConfirmingReset
        ) {
            Button("Reset File", role: .destructive) {
                Task { await model.resetSelectedDocument() }
            }
        }
    }

    private var workbenchToolbar: some View {
        HStack(spacing: 10) {
            if model.documents.count > 1 {
                Picker("Source file", selection: $model.selectedDocumentID) {
                    ForEach(model.documents) { document in
                        Text(document.displayName).tag(document.id)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 360)
            } else if let document = model.selectedDocument {
                Text(document.displayName)
                    .font(.caption.monospaced())
                    .lineLimit(1)
            }

            Spacer(minLength: 12)

            if model.supportedStages.count > 1 {
                Picker("Check stage", selection: $model.selectedStage) {
                    ForEach(model.supportedStages, id: \.self) { stage in
                        Text(stage.label).tag(stage)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 150)
            } else if let stage = model.supportedStages.first {
                Text(stage.label)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

            WorkbenchStatusLabel(phase: model.phase)

            Button {
                model.run()
            } label: {
                Image(systemName: "play.fill")
            }
            .disabled(!model.canRun)
            .help("Run selected check")
            .keyboardShortcut(.return, modifiers: .command)

            Button {
                model.cancel()
            } label: {
                Image(systemName: "stop.fill")
            }
            .disabled(!model.isRunning)
            .help("Stop the current run")

            Button {
                isConfirmingReset = true
            } label: {
                Image(systemName: "arrow.counterclockwise")
            }
            .disabled(model.selectedDocument == nil || model.isRunning)
            .help("Reset current file")

            Divider()
                .frame(height: 16)

            Button(action: onCollapse) {
                Image(systemName: placement.collapseSymbol)
            }
            .accessibilityLabel("Minimize workbench")
            .help("Minimize the code and build panel")
        }
        .controlSize(.small)
        .padding(.horizontal, 12)
        .frame(height: 42)
    }

    @ViewBuilder
    private var workbenchContent: some View {
        if workspaceAuthorization.authorization == nil {
            ContentUnavailableView {
                Label("Choose a build folder", systemImage: "folder.badge.plus")
            } description: {
                Text(workspaceAuthorization.errorMessage ?? """
                    LeetLLM needs one dedicated folder for editable course files, compiler \
                    output, and learner executables. macOS will ask you to choose it.
                    """)
            } actions: {
                Button("Choose Build Folder…") {
                    workspaceAuthorization.chooseFolder()
                }
            }
        } else if let setupError = model.setupError {
            ContentUnavailableView {
                Label("Workbench unavailable", systemImage: "wrench.and.screwdriver")
            } description: {
                Text(setupError)
            } actions: {
                Button("Retry") {
                    guard let authorization = workspaceAuthorization.authorization else { return }
                    Task { await model.prepare(authorization: authorization, force: true) }
                }
                Button("Choose Different Folder…") {
                    workspaceAuthorization.chooseFolder()
                }
            }
        } else if model.documents.isEmpty {
            ProgressView("Preparing learner workspace")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            HSplitView {
                CodeEditorView(
                    text: Binding(
                        get: { model.selectedDocument?.text ?? "" },
                        set: { model.updateSelectedDocument($0) }
                    ),
                    documentID: model.selectedDocumentID,
                    language: model.selectedDocument?.language ?? "text",
                    textScale: textSize,
                    isEditable: !model.isRunning,
                    onRun: model.run,
                    onSave: model.save
                )
                .frame(minWidth: 420)

                WorkbenchResultsView(model: model)
                    .frame(minWidth: 260, idealWidth: 340, maxWidth: 480)
            }
        }
    }
}

private struct WorkbenchStatusLabel: View {
    let phase: LessonWorkbenchModel.Phase

    var body: some View {
        HStack(spacing: 5) {
            if phase.isActive {
                ProgressView()
                    .controlSize(.mini)
            } else {
                Image(systemName: phase.symbolName)
                    .foregroundStyle(phase.color)
            }
            Text(phase.label)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize()
    }
}

private struct WorkbenchResultsView: View {
    @ObservedObject var model: LessonWorkbenchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("RESULTS")
                .font(.caption2.weight(.bold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 14)
                .frame(height: 34)

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(model.reports, id: \.stageID) { report in
                        JudgeReportView(report: report)
                    }
                    ForEach(Array(model.diagnostics.enumerated()), id: \.offset) { _, diagnostic in
                        Label {
                            Text(diagnostic.message)
                                .textSelection(.enabled)
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                        }
                        .font(.caption)
                    }
                    if !model.output.isEmpty {
                        DisclosureGroup("Build output") {
                            Text(model.output)
                                .font(.caption.monospaced())
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.top, 6)
                        }
                        .font(.caption)
                    }
                    if model.reports.isEmpty,
                       model.diagnostics.isEmpty,
                       model.output.isEmpty
                    {
                        Text("Run a check to see judge results.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

private struct JudgeReportView: View {
    let report: RunJudgeReport

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Label(
                    report.isPassing ? "Passed" : "Failed",
                    systemImage: report.isPassing
                        ? "checkmark.circle.fill"
                        : "xmark.circle.fill"
                )
                .foregroundStyle(report.isPassing ? .green : .red)
                Spacer()
                Text("\(report.passedCaseCount)/\(report.totalCaseCount)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            .font(.subheadline.weight(.semibold))

            ForEach(report.failures, id: \.id) { failure in
                VStack(alignment: .leading, spacing: 2) {
                    Text(failure.name)
                        .font(.caption.weight(.semibold))
                    Text(failure.message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
    }
}

@MainActor
private final class LessonWorkbenchModel: ObservableObject {
    struct Document: Identifiable, Sendable {
        let id: String
        let relativePath: String
        let displayName: String
        let language: String
        var text: String
    }

    enum Phase: Equatable {
        case idle
        case preparing
        case saving
        case building
        case running(RunStageID)
        case passed
        case failed
        case cancelling

        var isActive: Bool {
            switch self {
            case .preparing, .saving, .building, .running, .cancelling: true
            case .idle, .passed, .failed: false
            }
        }

        var label: String {
            switch self {
            case .idle: "Ready"
            case .preparing: "Preparing"
            case .saving: "Saving"
            case .building: "Building"
            case let .running(stage): "Running \(stage.label)"
            case .passed: "Passed"
            case .failed: "Failed"
            case .cancelling: "Stopping"
            }
        }

        var symbolName: String {
            switch self {
            case .passed: "checkmark.circle.fill"
            case .failed: "xmark.circle.fill"
            default: "circle"
            }
        }

        var color: Color {
            switch self {
            case .passed: .green
            case .failed: .red
            default: .secondary
            }
        }
    }

    @Published var documents: [Document] = []
    @Published var selectedDocumentID = ""
    @Published var selectedStage: RunStageID
    @Published var phase: Phase = .idle
    @Published var reports: [RunJudgeReport] = []
    @Published var diagnostics: [RunnerDiagnostic] = []
    @Published var output = ""
    @Published var setupError: String?
    @Published private(set) var isRunning = false

    let supportedStages: [RunStageID]
    private let lesson: LessonDocument
    private let activity: RuntimeActivityDescriptor
    private var workspace: LearnerWorkspace?
    private var runTask: Task<Void, Never>?
    private var saveTask: Task<Void, Never>?
    private var hasPrepared = false
    private var workspaceAuthorization: WorkspaceAuthorization?

    init(lesson: LessonDocument, activity: RuntimeActivityDescriptor) {
        self.lesson = lesson
        self.activity = activity
        supportedStages = activity.supportedStages
        selectedStage = activity.supportedStages.first ?? .cpu
    }

    var selectedDocument: Document? {
        documents.first { $0.id == selectedDocumentID }
    }

    var canRun: Bool {
        workspace != nil
            && workspaceAuthorization != nil
            && !documents.isEmpty
            && !isRunning
    }

    func revokeWorkspaceAuthorization() {
        runTask?.cancel()
        saveTask?.cancel()
        workspace = nil
        workspaceAuthorization = nil
        documents = []
        selectedDocumentID = ""
        reports = []
        diagnostics = []
        output = ""
        setupError = nil
        hasPrepared = false
        if !isRunning {
            phase = .idle
        }
    }

    func prepare(authorization: WorkspaceAuthorization, force: Bool = false) async {
        guard force || !hasPrepared || workspaceAuthorization?.id != authorization.id else {
            return
        }
        if workspaceAuthorization?.id != authorization.id {
            let previousRunTask = runTask
            let previousSaveTask = saveTask
            previousRunTask?.cancel()
            previousSaveTask?.cancel()
            await previousRunTask?.value
            await previousSaveTask?.value
            workspace = nil
            documents = []
            selectedDocumentID = ""
        }
        workspaceAuthorization = authorization
        hasPrepared = true
        phase = .preparing
        setupError = nil
        do {
            guard let sourceRoot = LearnerWorkspace.findSourceRoot(
                containing: lesson.sourceURL
            ) else {
                throw LearnerWorkspaceError.sourcePackageMissing(lesson.sourceURL)
            }
            let workspaceRoot = Self.workspaceRoot(
                for: sourceRoot,
                authorizationRoot: authorization.rootURL
            )
            let relativePaths = activity.exerciseFiles
                + [activity.metalFile].compactMap { $0 }
            let prepared = try await Task.detached {
                let workspace = try LearnerWorkspace.prepare(
                    sourceRoot: sourceRoot,
                    workspaceRoot: workspaceRoot
                )
                let documents = try relativePaths.map { relativePath in
                    Document(
                        id: relativePath,
                        relativePath: relativePath,
                        displayName: URL(fileURLWithPath: relativePath).lastPathComponent,
                        language: relativePath.hasSuffix(".metal") ? "metal" : "swift",
                        text: try workspace.read(relativePath)
                    )
                }
                return (workspace, documents)
            }.value
            workspace = prepared.0
            documents = prepared.1
            selectedDocumentID = documents.first?.id ?? ""
            phase = .idle
        } catch {
            setupError = error.localizedDescription
            phase = .failed
            hasPrepared = false
        }
    }

    func updateSelectedDocument(_ text: String) {
        guard let index = documents.firstIndex(where: { $0.id == selectedDocumentID }),
              documents[index].text != text
        else { return }
        documents[index].text = text
        scheduleSave()
    }

    func save() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            await self?.saveDocuments(showPhase: true)
        }
    }

    func run() {
        guard canRun else { return }
        isRunning = true
        runTask = Task { [weak self] in
            await self?.performRun()
        }
    }

    func cancel() {
        guard let runTask else { return }
        phase = .cancelling
        runTask.cancel()
    }

    func resetSelectedDocument() async {
        guard !isRunning,
              let workspace,
              let index = documents.firstIndex(where: { $0.id == selectedDocumentID })
        else { return }
        do {
            let relativePath = documents[index].relativePath
            let text = try await Task.detached {
                try workspace.reset(relativePath)
            }.value
            documents[index].text = text
            phase = .idle
        } catch {
            setupError = error.localizedDescription
            phase = .failed
        }
    }

    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            await self?.saveDocuments(showPhase: false)
        }
    }

    private func saveDocuments(showPhase: Bool) async {
        guard let workspace else { return }
        let snapshots = documents.map { ($0.relativePath, $0.text) }
        if showPhase, !isRunning {
            phase = .saving
        }
        do {
            try await Task.detached {
                for (relativePath, text) in snapshots {
                    try workspace.write(text, to: relativePath)
                }
            }.value
            if showPhase, !isRunning {
                phase = .idle
            }
        } catch {
            setupError = error.localizedDescription
            phase = .failed
        }
    }

    private func performRun() async {
        defer {
            runTask = nil
            isRunning = false
        }
        guard let workspace, let workspaceAuthorization else { return }
        reports = []
        diagnostics = []
        output = ""
        await saveDocuments(showPhase: false)
        guard !Task.isCancelled else {
            phase = .idle
            return
        }

        do {
            let client = try LocalRunnerClient.locate()
            let request = RunRequest(
                lessonID: lesson.id,
                activityID: activity.id,
                workspace: workspace.rootURL.path,
                workspaceBookmark: workspaceAuthorization.bookmarkData,
                mode: .debug,
                limits: RunLimits(
                    timeoutMilliseconds: 30_000,
                    maximumOutputBytes: 1_048_576,
                    maximumArtifactBytes: 8_388_608
                ),
                stages: [selectedStage],
                implementation: .learner
            )
            for try await event in client.events(for: request) {
                apply(event.event)
            }
        } catch is CancellationError {
            phase = .idle
        } catch {
            diagnostics.append(RunnerDiagnostic(
                severity: .error,
                code: "runner-client-failed",
                message: error.localizedDescription
            ))
            phase = .failed
        }
    }

    private func apply(_ event: RunEventPayload) {
        switch event {
        case .accepted:
            break
        case .buildStarted:
            phase = .building
        case let .buildFinished(result):
            if !result.succeeded {
                phase = .failed
            }
        case let .stageStarted(stage):
            phase = .running(stage.id)
        case let .judgeReport(report):
            reports.removeAll { $0.stageID == report.stageID }
            reports.append(report)
        case let .diagnostic(diagnostic):
            diagnostics.append(diagnostic)
        case let .stdout(stream), let .stderr(stream):
            output += stream.text
        case let .completed(completion):
            phase = completion.status == .passed ? .passed : .failed
        }
    }

    private static func workspaceRoot(for sourceRoot: URL, authorizationRoot: URL) -> URL {
        let digest = SHA256.hash(data: Data(sourceRoot.path.utf8))
            .prefix(8)
            .map { String(format: "%02x", $0) }
            .joined()
        return authorizationRoot
            .appending(path: "LeetLLM Workspaces", directoryHint: .isDirectory)
            .appending(path: digest, directoryHint: .isDirectory)
    }
}

private extension RunStageID {
    var label: String {
        switch self {
        case .cpu: "CPU"
        case .metal: "Metal"
        }
    }
}
