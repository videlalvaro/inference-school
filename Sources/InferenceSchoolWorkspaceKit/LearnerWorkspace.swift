import Foundation

public enum LearnerWorkspaceError: Error, LocalizedError, Equatable {
    case sourcePackageMissing(URL)
    case incompatibleWorkspace(URL)
    case unsafeRelativePath(String)
    case sourceFileMissing(String)

    public var errorDescription: String? {
        switch self {
        case let .sourcePackageMissing(url):
            "No Inference School source package was found at \(url.path)."
        case let .incompatibleWorkspace(url):
            "The learner workspace at \(url.path) uses an incompatible format."
        case let .unsafeRelativePath(path):
            "The workspace path is not a safe relative path: \(path)"
        case let .sourceFileMissing(path):
            "The course source file does not exist: \(path)"
        }
    }
}

public struct LearnerWorkspace: Sendable, Equatable {
    public static let schemaVersion = 1

    public let rootURL: URL
    public let sourceRootURL: URL

    public init(rootURL: URL, sourceRootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
        self.sourceRootURL = sourceRootURL.standardizedFileURL
    }

    public static func prepare(sourceRoot: URL, workspaceRoot: URL) throws -> Self {
        let sourceRoot = sourceRoot.standardizedFileURL
        let workspaceRoot = workspaceRoot.standardizedFileURL
        try validateSourceRoot(sourceRoot)

        if FileManager.default.fileExists(atPath: workspaceRoot.path) {
            let metadata = try loadMetadata(at: workspaceRoot)
            guard metadata.schemaVersion == schemaVersion,
                  metadata.sourceRoot == sourceRoot.path
            else {
                throw LearnerWorkspaceError.incompatibleWorkspace(workspaceRoot)
            }
            return Self(rootURL: workspaceRoot, sourceRootURL: sourceRoot)
        }

        let stagingRoot = workspaceRoot.deletingLastPathComponent().appending(
            path: ".\(workspaceRoot.lastPathComponent)-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: stagingRoot.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        do {
            try fileManager.createDirectory(at: stagingRoot, withIntermediateDirectories: true)
            for relativePath in sourceDirectories {
                let sourceURL = sourceRoot.appending(path: relativePath, directoryHint: .isDirectory)
                let destinationURL = stagingRoot.appending(
                    path: relativePath,
                    directoryHint: .isDirectory
                )
                try fileManager.createDirectory(
                    at: destinationURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try fileManager.copyItem(at: sourceURL, to: destinationURL)
            }
            try packageManifest.write(
                to: stagingRoot.appending(path: "Package.swift"),
                atomically: true,
                encoding: .utf8
            )
            let metadata = WorkspaceMetadata(
                schemaVersion: schemaVersion,
                sourceRoot: sourceRoot.path
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(metadata).write(
                to: stagingRoot.appending(path: metadataFilename),
                options: .atomic
            )
            try fileManager.moveItem(at: stagingRoot, to: workspaceRoot)
        } catch {
            try? fileManager.removeItem(at: stagingRoot)
            throw error
        }
        return Self(rootURL: workspaceRoot, sourceRootURL: sourceRoot)
    }

    public static func findSourceRoot(containing sourceURL: URL) -> URL? {
        var candidate = sourceURL.hasDirectoryPath
            ? sourceURL.standardizedFileURL
            : sourceURL.deletingLastPathComponent().standardizedFileURL
        while candidate.path != "/" {
            if isSourceRoot(candidate) {
                return candidate
            }
            candidate.deleteLastPathComponent()
        }
        return nil
    }

    public func fileURL(for relativePath: String) throws -> URL {
        try Self.safeURL(for: relativePath, under: rootURL)
    }

    public func read(_ relativePath: String) throws -> String {
        let url = try fileURL(for: relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    public func write(_ contents: String, to relativePath: String) throws {
        let url = try fileURL(for: relativePath)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    public func reset(_ relativePath: String) throws -> String {
        let sourceURL = try Self.safeURL(for: relativePath, under: sourceRootURL)
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw LearnerWorkspaceError.sourceFileMissing(relativePath)
        }
        let contents = try String(contentsOf: sourceURL, encoding: .utf8)
        try write(contents, to: relativePath)
        return contents
    }

    private struct WorkspaceMetadata: Codable {
        let schemaVersion: Int
        let sourceRoot: String
    }

    private static let metadataFilename = ".inference-school-workspace.json"
    private static let sourceDirectories = [
        "Sources/InferenceSchoolCore",
        "Sources/InferenceSchoolExercises",
        "Sources/InferenceSchoolSolutions",
        "Sources/InferenceSchoolRunnerProtocol",
        "Sources/InferenceSchoolCLI",
        "Sources/InferenceSchoolCLIEntry",
    ]

    private static func validateSourceRoot(_ sourceRoot: URL) throws {
        guard isSourceRoot(sourceRoot),
              sourceDirectories.allSatisfy({ relativePath in
                  FileManager.default.fileExists(
                      atPath: sourceRoot.appending(path: relativePath).path
                  )
              })
        else {
            throw LearnerWorkspaceError.sourcePackageMissing(sourceRoot)
        }
    }

    private static func isSourceRoot(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appending(path: "Package.swift").path)
            && FileManager.default.fileExists(
                atPath: url.appending(path: "Sources/InferenceSchoolCore").path
            )
    }

    private static func loadMetadata(at workspaceRoot: URL) throws -> WorkspaceMetadata {
        let url = workspaceRoot.appending(path: metadataFilename)
        guard let data = try? Data(contentsOf: url),
              let metadata = try? JSONDecoder().decode(WorkspaceMetadata.self, from: data)
        else {
            throw LearnerWorkspaceError.incompatibleWorkspace(workspaceRoot)
        }
        return metadata
    }

    private static func safeURL(for relativePath: String, under root: URL) throws -> URL {
        let path = NSString(string: relativePath).standardizingPath
        guard !relativePath.isEmpty,
              !relativePath.hasPrefix("/"),
              path != "..",
              !path.hasPrefix("../")
        else {
            throw LearnerWorkspaceError.unsafeRelativePath(relativePath)
        }
        let root = root.standardizedFileURL
        let resolved = root.appending(path: path).standardizedFileURL
        guard resolved.path.hasPrefix(root.path + "/") else {
            throw LearnerWorkspaceError.unsafeRelativePath(relativePath)
        }
        return resolved
    }

    private static let packageManifest = """
        // swift-tools-version: 6.2

        import PackageDescription

        let package = Package(
            name: "InferenceSchoolLearnerWorkspace",
            platforms: [.macOS(.v15)],
            products: [
                .executable(name: "inference-school", targets: ["InferenceSchoolCLI"]),
            ],
            targets: [
                .target(name: "InferenceSchoolCore"),
                .target(name: "InferenceSchoolRunnerProtocol"),
                .target(
                    name: "InferenceSchoolExercises",
                    dependencies: ["InferenceSchoolCore"],
                    resources: [.copy("Metal")]
                ),
                .target(
                    name: "InferenceSchoolSolutions",
                    dependencies: ["InferenceSchoolCore"],
                    resources: [.copy("Metal")]
                ),
                .target(
                    name: "InferenceSchoolRuntime",
                    dependencies: [
                        "InferenceSchoolCore",
                        "InferenceSchoolExercises",
                        "InferenceSchoolSolutions",
                        "InferenceSchoolRunnerProtocol",
                    ],
                    path: "Sources/InferenceSchoolCLI"
                ),
                .executableTarget(
                    name: "InferenceSchoolCLI",
                    dependencies: ["InferenceSchoolRuntime"],
                    path: "Sources/InferenceSchoolCLIEntry"
                ),
            ]
        )
        """
}