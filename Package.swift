// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "InferenceSchool",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "InferenceSchoolCore", targets: ["InferenceSchoolCore"]),
        .library(name: "InferenceSchoolLessonKit", targets: ["InferenceSchoolLessonKit"]),
        .library(name: "InferenceSchoolRunnerProtocol", targets: ["InferenceSchoolRunnerProtocol"]),
        .library(name: "InferenceSchoolRuntime", targets: ["InferenceSchoolRuntime"]),
        .library(name: "InferenceSchoolRunnerClient", targets: ["InferenceSchoolRunnerClient"]),
        .library(name: "InferenceSchoolWorkspaceKit", targets: ["InferenceSchoolWorkspaceKit"]),
        .executable(name: "inference-school", targets: ["InferenceSchoolCLI"]),
        .executable(name: "inference-school-runner", targets: ["InferenceSchoolRunner"]),
        .executable(name: "inference-school-studio", targets: ["InferenceSchoolStudio"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/gonzalezreal/textual",
            exact: "0.5.0"
        ),
        .package(
            url: "https://github.com/jpsim/Yams",
            exact: "6.2.2"
        ),
    ],
    targets: [
        .target(name: "InferenceSchoolCore"),
        .target(name: "InferenceSchoolRunnerProtocol"),
        .target(
            name: "InferenceSchoolRunnerClient",
            dependencies: ["InferenceSchoolRunnerProtocol"]
        ),
        .target(name: "InferenceSchoolWorkspaceKit"),
        .target(
            name: "InferenceSchoolLessonKit",
            dependencies: [
                .product(name: "Yams", package: "Yams"),
            ]
        ),
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
        .executableTarget(
            name: "InferenceSchoolRunner",
            dependencies: ["InferenceSchoolRuntime", "InferenceSchoolRunnerProtocol"],
            linkerSettings: [.linkedFramework("Security")]
        ),
        .executableTarget(
            name: "InferenceSchoolStudio",
            dependencies: [
                "InferenceSchoolLessonKit",
                "InferenceSchoolRuntime",
                "InferenceSchoolRunnerClient",
                "InferenceSchoolRunnerProtocol",
                "InferenceSchoolWorkspaceKit",
                .product(name: "Textual", package: "textual"),
            ],
            resources: [.copy("Resources")],
            linkerSettings: [.linkedFramework("Security")]
        ),
        .testTarget(
            name: "InferenceSchoolLessonKitTests",
            dependencies: [
                "InferenceSchoolLessonKit",
                .product(name: "Textual", package: "textual"),
            ]
        ),
        .testTarget(
            name: "InferenceSchoolRuntimeTests",
            dependencies: [
                "InferenceSchoolLessonKit",
                "InferenceSchoolCore",
                "InferenceSchoolRuntime",
                "InferenceSchoolRunnerProtocol",
            ]
        ),
        .testTarget(
            name: "InferenceSchoolWorkspaceKitTests",
            dependencies: ["InferenceSchoolWorkspaceKit"]
        ),
        .testTarget(
            name: "InferenceSchoolRunnerClientTests",
            dependencies: [
                "InferenceSchoolRunnerClient",
                "InferenceSchoolRunnerProtocol",
                "InferenceSchoolWorkspaceKit",
            ]
        ),
        .testTarget(
            name: "InferenceSchoolCoreTests",
            dependencies: ["InferenceSchoolCore", "InferenceSchoolSolutions"]
        ),
        .testTarget(
            name: "InferenceSchoolStudioTests",
            dependencies: ["InferenceSchoolStudio", "InferenceSchoolLessonKit"]
        ),
    ]
)