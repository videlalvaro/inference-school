import Darwin
import Foundation
import InferenceSchoolRuntime

do {
    let exitCode = try InferenceSchoolCommand.run(arguments: Array(CommandLine.arguments.dropFirst()))
    exit(exitCode)
} catch {
    print("error: \(error.localizedDescription)")
    exit(2)
}