import Darwin
import Foundation
import InferenceSchoolRunnerProtocol

struct SwiftCourseBuild {
    let result: ChildProcessResult
    let executableURL: URL
}

enum SwiftCourseBuilder {
    static func build(
        workspaceURL: URL,
        mode: RunMode,
        environment: [String: String],
        timeoutMilliseconds: Int?,
        maximumOutputBytes: Int
    ) throws -> SwiftCourseBuild {
        let fileManager = FileManager.default
        let workspaceURL = workspaceURL.resolvingSymlinksInPath()
        let buildRoot = workspaceURL.appending(
            path: ".inference-school-build",
            directoryHint: .isDirectory
        )
        try? fileManager.removeItem(at: buildRoot)

        let configuration = mode == .release ? "release" : "debug"
        let outputRoot = buildRoot.appending(path: configuration, directoryHint: .isDirectory)
        let modulesRoot = outputRoot.appending(path: "Modules", directoryHint: .isDirectory)
        let objectsRoot = outputRoot.appending(path: "Objects", directoryHint: .isDirectory)
        let moduleCacheRoot = outputRoot.appending(
            path: "ModuleCache",
            directoryHint: .isDirectory
        )
        for directory in [modulesRoot, objectsRoot, moduleCacheRoot] {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        let executableURL = outputRoot.appending(path: "inference-school")
        var commands = BuildCommandAccumulator(
            environment: environment,
            timeoutMilliseconds: timeoutMilliseconds,
            maximumOutputBytes: maximumOutputBytes
        )

        let developerDirectory: URL
        if let configured = environment["DEVELOPER_DIR"], !configured.isEmpty {
            developerDirectory = URL(fileURLWithPath: configured, isDirectory: true)
        } else {
            let selection = try commands.run(
                executableURL: URL(fileURLWithPath: "/usr/bin/xcode-select"),
                arguments: ["--print-path"]
            )
            guard commands.shouldContinue(after: selection) else {
                return commands.finish(executableURL: executableURL, lastResult: selection)
            }
            let path = selection.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else {
                throw SwiftCourseBuilderError.developerDirectoryMissing
            }
            developerDirectory = URL(fileURLWithPath: path, isDirectory: true)
        }

        let swiftCompiler = developerDirectory.appending(
            path: "Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc"
        )
        guard fileManager.isExecutableFile(atPath: swiftCompiler.path) else {
            throw SwiftCourseBuilderError.swiftCompilerMissing(swiftCompiler)
        }

        let sdkRoot: URL
        if let configured = environment["SDKROOT"], !configured.isEmpty {
            sdkRoot = URL(fileURLWithPath: configured, isDirectory: true)
        } else {
            sdkRoot = developerDirectory.appending(
                path: "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
                directoryHint: .isDirectory
            )
        }
        guard fileManager.fileExists(atPath: sdkRoot.path) else {
            throw SwiftCourseBuilderError.sdkMissing(sdkRoot)
        }

        let target = "\(architecture)-apple-macosx15.0"
        var commonArguments = [
            "-swift-version", "6",
            "-target", target,
            "-sdk", sdkRoot.path,
            "-module-cache-path", moduleCacheRoot.path,
            "-I", modulesRoot.path,
            "-DSWIFT_PACKAGE",
        ]
        commonArguments += mode == .release ? ["-O"] : ["-Onone", "-g"]

        let exerciseAccessor = outputRoot.appending(path: "InferenceSchoolExercises-Bundle.swift")
        let solutionAccessor = outputRoot.appending(path: "InferenceSchoolSolutions-Bundle.swift")
        try resourceAccessor(moduleName: "InferenceSchoolExercises").write(
            to: exerciseAccessor,
            atomically: true,
            encoding: .utf8
        )
        try resourceAccessor(moduleName: "InferenceSchoolSolutions").write(
            to: solutionAccessor,
            atomically: true,
            encoding: .utf8
        )

        let modules = [
            Module(
                name: "InferenceSchoolCore",
                sourceDirectory: "Sources/InferenceSchoolCore"
            ),
            Module(
                name: "InferenceSchoolRunnerProtocol",
                sourceDirectory: "Sources/InferenceSchoolRunnerProtocol"
            ),
            Module(
                name: "InferenceSchoolExercises",
                sourceDirectory: "Sources/InferenceSchoolExercises",
                additionalSources: [exerciseAccessor]
            ),
            Module(
                name: "InferenceSchoolSolutions",
                sourceDirectory: "Sources/InferenceSchoolSolutions",
                additionalSources: [solutionAccessor]
            ),
            Module(
                name: "InferenceSchoolRuntime",
                sourceDirectory: "Sources/InferenceSchoolCLI"
            ),
        ]

        var objectURLs: [URL] = []
        for module in modules {
            let sourceRoot = workspaceURL.appending(
                path: module.sourceDirectory,
                directoryHint: .isDirectory
            )
            let sources = try swiftSources(under: sourceRoot, workspaceURL: workspaceURL)
                + module.additionalSources
            guard !sources.isEmpty else {
                throw SwiftCourseBuilderError.sourceDirectoryEmpty(sourceRoot)
            }

            let objectURL = objectsRoot.appending(path: "\(module.name).o")
            let moduleURL = modulesRoot.appending(path: "\(module.name).swiftmodule")
            let sourceArguments = sources.map { $0.path }
            let result = try commands.run(
                executableURL: swiftCompiler,
                arguments: commonArguments + [
                    "-module-name", module.name,
                    "-parse-as-library",
                    "-whole-module-optimization",
                    "-emit-object",
                    "-emit-module",
                    "-emit-module-path", moduleURL.path,
                    "-o", objectURL.path,
                ] + sourceArguments
            )
            guard commands.shouldContinue(after: result) else {
                return commands.finish(executableURL: executableURL, lastResult: result)
            }
            objectURLs.append(objectURL)
        }

        let entryPoint = workspaceURL.appending(path: "Sources/InferenceSchoolCLIEntry/main.swift")
        guard fileManager.fileExists(atPath: entryPoint.path) else {
            throw SwiftCourseBuilderError.sourceFileMissing(entryPoint)
        }
        let linkArguments = commonArguments + [
            "-module-name", "InferenceSchoolCLI",
            "-emit-executable",
            entryPoint.path,
        ] + objectURLs.map { $0.path } + ["-o", executableURL.path]
        let linkResult = try commands.run(
            executableURL: swiftCompiler,
            arguments: linkArguments
        )
        guard commands.shouldContinue(after: linkResult) else {
            return commands.finish(executableURL: executableURL, lastResult: linkResult)
        }

        if SandboxProcess.isAppSandboxed {
            let entitlementsURL = outputRoot.appending(path: "Learner.entitlements")
            try learnerEntitlements.write(
                to: entitlementsURL,
                atomically: true,
                encoding: .utf8
            )
            let signingResult = try commands.run(
                executableURL: URL(fileURLWithPath: "/usr/bin/codesign"),
                arguments: [
                    "--force",
                    "--sign", "-",
                    "--entitlements", entitlementsURL.path,
                    "--timestamp=none",
                    executableURL.path,
                ]
            )
            guard commands.shouldContinue(after: signingResult) else {
                return commands.finish(
                    executableURL: executableURL,
                    lastResult: signingResult
                )
            }
        }

        try copyResources(
            named: "InferenceSchoolExercises",
            from: workspaceURL.appending(
                path: "Sources/InferenceSchoolExercises/Metal",
                directoryHint: .isDirectory
            ),
            to: outputRoot
        )
        try copyResources(
            named: "InferenceSchoolSolutions",
            from: workspaceURL.appending(
                path: "Sources/InferenceSchoolSolutions/Metal",
                directoryHint: .isDirectory
            ),
            to: outputRoot
        )

        return commands.finish(executableURL: executableURL, lastResult: linkResult)
    }

    private struct Module {
        let name: String
        let sourceDirectory: String
        var additionalSources: [URL] = []
    }

    private static var architecture: String {
        #if arch(arm64)
        "arm64"
        #elseif arch(x86_64)
        "x86_64"
        #else
        #error("Unsupported macOS architecture")
        #endif
    }

    private static func swiftSources(under root: URL, workspaceURL: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw SwiftCourseBuilderError.sourceDirectoryMissing(root)
        }
        let resolvedWorkspace = workspaceURL.resolvingSymlinksInPath()
        return try enumerator.compactMap { element in
            guard let url = element as? URL, url.pathExtension == "swift" else {
                return nil
            }
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { return nil }
            let resolved = url.resolvingSymlinksInPath()
            guard resolved.path.hasPrefix(resolvedWorkspace.path + "/") else {
                throw SwiftCourseBuilderError.sourceOutsideWorkspace(url)
            }
            return resolved
        }.sorted { $0.path < $1.path }
    }

    private static func resourceAccessor(moduleName: String) -> String {
        """
        import Foundation

        extension Foundation.Bundle {
            static nonisolated let module: Bundle = {
                let path = Bundle.main.bundleURL
                    .appendingPathComponent("InferenceSchool_\(moduleName).bundle")
                    .path
                guard let bundle = Bundle(path: path) else {
                    Swift.fatalError("could not load resource bundle at \\(path)")
                }
                return bundle
            }()
        }
        """
    }

    private static let learnerEntitlements = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
        "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>com.apple.security.app-sandbox</key>
            <true/>
            <key>com.apple.security.inherit</key>
            <true/>
        </dict>
        </plist>
        """

    private static func copyResources(named moduleName: String, from source: URL, to root: URL)
        throws
    {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: source.path) else {
            throw SwiftCourseBuilderError.sourceDirectoryMissing(source)
        }
        let bundle = root.appending(
            path: "InferenceSchool_\(moduleName).bundle",
            directoryHint: .isDirectory
        )
        try fileManager.createDirectory(at: bundle, withIntermediateDirectories: true)
        try fileManager.copyItem(
            at: source,
            to: bundle.appending(path: "Metal", directoryHint: .isDirectory)
        )
    }
}

private struct BuildCommandAccumulator {
    let environment: [String: String]
    let timeoutMilliseconds: Int?
    let maximumOutputBytes: Int
    let startedAt = DispatchTime.now().uptimeNanoseconds
    var stdout = ""
    var stderr = ""
    var outputLimitExceeded = false
    var timedOut = false

    mutating func run(executableURL: URL, arguments: [String]) throws -> ChildProcessResult {
        let elapsedMilliseconds = Int(
            (DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000
        )
        let remainingTimeout = timeoutMilliseconds.map { max(0, $0 - elapsedMilliseconds) }
        guard remainingTimeout != 0 else {
            let result = ChildProcessResult(
                exitCode: -1,
                stdout: "",
                stderr: "",
                stdoutTruncated: false,
                stderrTruncated: false,
                outputLimitExceeded: false,
                timedOut: true
            )
            timedOut = true
            return result
        }

        let usedBytes = stdout.utf8.count + stderr.utf8.count
        let remainingOutputBytes = max(1, maximumOutputBytes - usedBytes)
        let result = try ChildProcess.run(
            executableURL: executableURL,
            arguments: arguments,
            environment: environment,
            timeoutMilliseconds: remainingTimeout,
            maximumOutputBytes: remainingOutputBytes
        )
        stdout += result.stdout
        stderr += result.stderr
        outputLimitExceeded = outputLimitExceeded || result.outputLimitExceeded
            || stdout.utf8.count + stderr.utf8.count > maximumOutputBytes
        timedOut = timedOut || result.timedOut
        return result
    }

    func shouldContinue(after result: ChildProcessResult) -> Bool {
        result.exitCode == 0 && !timedOut && !outputLimitExceeded
    }

    func finish(executableURL: URL, lastResult: ChildProcessResult) -> SwiftCourseBuild {
        SwiftCourseBuild(
            result: ChildProcessResult(
                exitCode: lastResult.exitCode,
                stdout: stdout,
                stderr: stderr,
                stdoutTruncated: outputLimitExceeded,
                stderrTruncated: outputLimitExceeded,
                outputLimitExceeded: outputLimitExceeded,
                timedOut: timedOut
            ),
            executableURL: executableURL
        )
    }
}

private enum SwiftCourseBuilderError: Error, LocalizedError {
    case developerDirectoryMissing
    case swiftCompilerMissing(URL)
    case sdkMissing(URL)
    case sourceDirectoryMissing(URL)
    case sourceDirectoryEmpty(URL)
    case sourceFileMissing(URL)
    case sourceOutsideWorkspace(URL)

    var errorDescription: String? {
        switch self {
        case .developerDirectoryMissing:
            "xcode-select did not return an active developer directory."
        case let .swiftCompilerMissing(url):
            "The Swift compiler was not found at \(url.path)."
        case let .sdkMissing(url):
            "The macOS SDK was not found at \(url.path)."
        case let .sourceDirectoryMissing(url):
            "The learner source directory is missing: \(url.path)"
        case let .sourceDirectoryEmpty(url):
            "The learner source directory has no Swift files: \(url.path)"
        case let .sourceFileMissing(url):
            "The learner source file is missing: \(url.path)"
        case let .sourceOutsideWorkspace(url):
            "A learner source resolves outside the workspace: \(url.path)"
        }
    }
}