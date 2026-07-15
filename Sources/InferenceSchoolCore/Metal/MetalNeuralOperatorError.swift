import Foundation

public enum MetalNeuralOperatorError: Error, LocalizedError {
    case noDevice
    case commandQueueCreationFailed
    case libraryCreationFailed(operation: String, message: String)
    case functionNotFound(String)
    case pipelineCreationFailed(operation: String, message: String)
    case unsupportedThreadgroupWidth(required: Int, maximum: Int)
    case dimensionsTooLarge
    case rowWidthExceedsMaximum(maximum: Int, actual: Int)
    case bufferCreationFailed(String)
    case commandCreationFailed
    case commandFailed(operation: String, message: String)
    case kernelResourceMissing(String)

    public var errorDescription: String? {
        switch self {
        case .noDevice: "Metal is unavailable on this machine."
        case .commandQueueCreationFailed: "Metal could not create a command queue."
        case let .libraryCreationFailed(operation, message): "Metal could not compile the \(operation) library: \(message)"
        case let .functionNotFound(name): "The Metal library does not contain \(name)."
        case let .pipelineCreationFailed(operation, message): "Metal could not create the \(operation) pipeline: \(message)"
        case let .unsupportedThreadgroupWidth(required, maximum): "The kernel requires \(required) threads per group; the pipeline supports \(maximum)."
        case .dimensionsTooLarge: "Operator dimensions exceed UInt32.max."
        case let .rowWidthExceedsMaximum(maximum, actual): "Softmax row width must not exceed \(maximum); received \(actual)."
        case let .bufferCreationFailed(name): "Metal could not allocate the \(name) buffer."
        case .commandCreationFailed: "Metal could not create a command buffer or encoder."
        case let .commandFailed(operation, message): "The Metal \(operation) command failed: \(message)"
        case let .kernelResourceMissing(name): "The \(name) Metal source resource is missing."
        }
    }
}