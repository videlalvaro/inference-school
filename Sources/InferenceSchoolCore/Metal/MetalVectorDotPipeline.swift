import Foundation
import Metal

public enum MetalVectorDotError: Error, LocalizedError {
    case noDevice
    case commandQueueCreationFailed
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case unsupportedThreadgroupWidth(maximum: Int)
    case inputTooLarge(Int)
    case bufferCreationFailed(String)
    case commandCreationFailed
    case commandFailed(String)
    case kernelResourceMissing(String)

    public var errorDescription: String? {
        switch self {
        case .noDevice:
            "Metal is unavailable on this machine."
        case .commandQueueCreationFailed:
            "Metal could not create a command queue."
        case let .functionNotFound(name):
            "The Metal library does not contain a \(name) function."
        case let .pipelineCreationFailed(message):
            "Metal could not create the compute pipeline: \(message)"
        case let .unsupportedThreadgroupWidth(maximum):
            "The kernel requires 256 threads per threadgroup, but this device supports \(maximum)."
        case let .inputTooLarge(count):
            "The vector contains \(count) elements; this lab supports at most \(UInt32.max)."
        case let .bufferCreationFailed(name):
            "Metal could not allocate the \(name) buffer."
        case .commandCreationFailed:
            "Metal could not create a command buffer or compute encoder."
        case let .commandFailed(message):
            "The Metal command failed: \(message)"
        case let .kernelResourceMissing(name):
            "The \(name) Metal source resource is missing."
        }
    }
}

public final class MetalVectorDotPipeline {
    public static let threadgroupWidth = 256

    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "vector_dot_partial") throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalVectorDotError.noDevice
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalVectorDotError.commandQueueCreationFailed
        }

        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalVectorDotError.pipelineCreationFailed(error.localizedDescription)
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw MetalVectorDotError.functionNotFound(functionName)
        }

        let pipeline: any MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalVectorDotError.pipelineCreationFailed(error.localizedDescription)
        }

        guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else {
            throw MetalVectorDotError.unsupportedThreadgroupWidth(
                maximum: pipeline.maxTotalThreadsPerThreadgroup
            )
        }

        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pipeline
    }

    public func dot(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
        guard lhs.count == rhs.count else {
            throw VectorDotError.lengthMismatch(lhs: lhs.count, rhs: rhs.count)
        }
        guard lhs.count <= UInt32.max else {
            throw MetalVectorDotError.inputTooLarge(lhs.count)
        }
        guard !lhs.isEmpty else {
            return 0
        }

        let inputByteCount = lhs.count * MemoryLayout<Float>.stride
        guard let lhsBuffer = device.makeBuffer(
            bytes: lhs,
            length: inputByteCount,
            options: .storageModeShared
        ) else {
            throw MetalVectorDotError.bufferCreationFailed("left input")
        }
        guard let rhsBuffer = device.makeBuffer(
            bytes: rhs,
            length: inputByteCount,
            options: .storageModeShared
        ) else {
            throw MetalVectorDotError.bufferCreationFailed("right input")
        }

        let groupCount = (lhs.count + Self.threadgroupWidth - 1) / Self.threadgroupWidth
        let partialByteCount = groupCount * MemoryLayout<Float>.stride
        guard let partialBuffer = device.makeBuffer(
            length: partialByteCount,
            options: .storageModeShared
        ) else {
            throw MetalVectorDotError.bufferCreationFailed("partial sum")
        }
        guard
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw MetalVectorDotError.commandCreationFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
        encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
        encoder.setBuffer(partialBuffer, offset: 0, index: 2)
        var elementCount = UInt32(lhs.count)
        encoder.setBytes(
            &elementCount,
            length: MemoryLayout<UInt32>.stride,
            index: 3
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: groupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: Self.threadgroupWidth,
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        guard commandBuffer.status == .completed else {
            throw MetalVectorDotError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "unknown error"
            )
        }

        let partials = partialBuffer.contents().bindMemory(
            to: Float.self,
            capacity: groupCount
        )
        var result: Float = 0
        for index in 0..<groupCount {
            result += partials[index]
        }
        return result
    }
}