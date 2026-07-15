import Foundation
import Security

final class SecurityScopedWorkspaceAccess {
    private let rootURL: URL
    private var isActive: Bool

    private init(rootURL: URL, isActive: Bool) {
        self.rootURL = rootURL
        self.isActive = isActive
    }

    static func acquire(
        bookmarkData: Data?,
        workspaceURL: URL
    ) throws -> SecurityScopedWorkspaceAccess? {
        guard let bookmarkData else {
            guard !SandboxProcess.isAppSandboxed else {
                throw SecurityScopedWorkspaceError.bookmarkRequired
            }
            return nil
        }

        let rootURL: URL
        do {
            var isStale = false
            rootURL = try URL(
                resolvingBookmarkData: bookmarkData,
                options: [.withSecurityScope, .withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
        } catch {
            throw SecurityScopedWorkspaceError.invalidBookmark(error.localizedDescription)
        }

        let didStart = rootURL.startAccessingSecurityScopedResource()
        guard didStart || !SandboxProcess.isAppSandboxed else {
            throw SecurityScopedWorkspaceError.accessDenied
        }

        let access = SecurityScopedWorkspaceAccess(rootURL: rootURL, isActive: didStart)
        guard Self.contains(workspaceURL, under: rootURL) else {
            access.stop()
            throw SecurityScopedWorkspaceError.workspaceOutsideSelectedFolder
        }
        return access
    }

    func stop() {
        guard isActive else { return }
        rootURL.stopAccessingSecurityScopedResource()
        isActive = false
    }

    deinit {
        stop()
    }

    private static func contains(_ candidate: URL, under root: URL) -> Bool {
        let candidate = candidate.standardizedFileURL.resolvingSymlinksInPath().path
        let root = root.standardizedFileURL.resolvingSymlinksInPath().path
        return candidate == root || candidate.hasPrefix(root + "/")
    }
}

enum SandboxProcess {
    static let isAppSandboxed: Bool = {
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

private enum SecurityScopedWorkspaceError: Error, LocalizedError {
    case bookmarkRequired
    case invalidBookmark(String)
    case accessDenied
    case workspaceOutsideSelectedFolder

    var errorDescription: String? {
        switch self {
        case .bookmarkRequired:
            "Choose a build folder in Inference School Studio before running learner code."
        case let .invalidBookmark(message):
            "The saved build-folder permission is invalid: \(message)"
        case .accessDenied:
            "macOS did not grant access to the selected build folder. Choose it again."
        case .workspaceOutsideSelectedFolder:
            "The learner workspace is outside the folder authorized by macOS."
        }
    }
}