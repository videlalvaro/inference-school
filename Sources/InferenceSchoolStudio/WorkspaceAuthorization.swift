import AppKit
import Foundation
import Security

struct WorkspaceAuthorization: Identifiable, Sendable {
    let id = UUID()
    let rootURL: URL
    let bookmarkData: Data
}

@MainActor
final class WorkspaceAuthorizationController: ObservableObject {
    @Published private(set) var authorization: WorkspaceAuthorization?
    @Published private(set) var errorMessage: String?

    private static let bookmarkKey = "studio.workspaceBookmark"
    private var activeURL: URL?

    init() {
        restoreBookmark()
    }

    deinit {
        activeURL?.stopAccessingSecurityScopedResource()
    }

    var selectedFolderName: String? {
        authorization?.rootURL.lastPathComponent
    }

    func chooseFolder() {
        let panel = NSOpenPanel()
        panel.title = "Choose a Inference School Build Folder"
        panel.message = """
            Choose or create a dedicated folder. Inference School will store editable course files, \
            compiler output, and learner executables inside it. Learner code runs without \
            network access in the sandboxed app.
            """
        panel.prompt = "Use Build Folder"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first

        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let bookmarkData = try url.bookmarkData(
                options: .withSecurityScope,
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            try activate(url: url, bookmarkData: bookmarkData)
            UserDefaults.standard.set(bookmarkData, forKey: Self.bookmarkKey)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func forgetFolder() {
        activeURL?.stopAccessingSecurityScopedResource()
        activeURL = nil
        authorization = nil
        errorMessage = nil
        UserDefaults.standard.removeObject(forKey: Self.bookmarkKey)
    }

    private func restoreBookmark() {
        guard let bookmarkData = UserDefaults.standard.data(forKey: Self.bookmarkKey) else {
            return
        }
        do {
            var isStale = false
            let url = try URL(
                resolvingBookmarkData: bookmarkData,
                options: [.withSecurityScope, .withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            var currentBookmark = bookmarkData
            if isStale {
                currentBookmark = try url.bookmarkData(
                    options: .withSecurityScope,
                    includingResourceValuesForKeys: nil,
                    relativeTo: nil
                )
                UserDefaults.standard.set(currentBookmark, forKey: Self.bookmarkKey)
            }
            try activate(url: url, bookmarkData: currentBookmark)
        } catch {
            errorMessage = "Choose the build folder again. \(error.localizedDescription)"
        }
    }

    private func activate(url: URL, bookmarkData: Data) throws {
        let didStart = url.startAccessingSecurityScopedResource()
        guard didStart || !AppSandbox.isEnabled else {
            throw WorkspaceAuthorizationError.accessDenied
        }
        activeURL?.stopAccessingSecurityScopedResource()
        activeURL = didStart ? url : nil
        authorization = WorkspaceAuthorization(
            rootURL: url.standardizedFileURL,
            bookmarkData: bookmarkData
        )
    }
}

private enum AppSandbox {
    static let isEnabled: Bool = {
        guard let task = SecTaskCreateFromSelf(nil),
              let value = SecTaskCopyValueForEntitlement(
                task,
                "com.apple.security.app-sandbox" as CFString,
                nil
              ) as? NSNumber
        else {
            return false
        }
        return value.boolValue
    }()
}

private enum WorkspaceAuthorizationError: Error, LocalizedError {
    case accessDenied

    var errorDescription: String? {
        "macOS did not grant access to that folder. Choose it again."
    }
}