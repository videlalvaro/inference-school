import Foundation
import Metal

public enum MetalTransposeError: Error, LocalizedError {
    case noDevice
    case commandQueueCreationFailed
    case libraryCreationFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case dimensionsTooLarge
    case bufferCreationFailed(String)
    case commandCreationFailed
    case commandFailed(String)
    case kernelResourceMissing(String)

    public var errorDescription: String? {
        switch self {
        case .noDevice: "Metal is unavailable on this machine."
        case .commandQueueCreationFailed: "Metal could not create a command queue."
        case let .libraryCreationFailed(message): "Metal could not compile the transpose library: \(message)"
        case let .functionNotFound(name): "The Metal library does not contain \(name)."
        case let .pipelineCreationFailed(message): "Metal could not create the transpose pipeline: \(message)"
        case .dimensionsTooLarge: "Matrix dimensions exceed UInt32.max."
        case let .bufferCreationFailed(name): "Metal could not allocate the \(name) buffer."
        case .commandCreationFailed: "Metal could not create a command buffer or encoder."
        case let .commandFailed(message): "The Metal transpose command failed: \(message)"
        case let .kernelResourceMissing(name): "The \(name) Metal source resource is missing."
        }
    }
}

public final class MetalTransposePipeline {
    public static let tileWidth = 16

    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "tiled_transpose") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalTransposeError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalTransposeError.commandQueueCreationFailed
        }

        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalTransposeError.libraryCreationFailed(error.localizedDescription)
        }
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalTransposeError.functionNotFound(functionName)
        }
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalTransposeError.pipelineCreationFailed(error.localizedDescription)
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func transpose(_ input: FloatTensor) throws -> FloatTensor {
        guard input.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: input.rank)
        }
        let rows = input.shape[0]
        let columns = input.shape[1]
        guard rows <= UInt32.max, columns <= UInt32.max else {
            throw MetalTransposeError.dimensionsTooLarge
        }
        guard !input.storage.isEmpty else {
            return try FloatTensor([], shape: [columns, rows])
        }

        let byteCount = input.storage.count * MemoryLayout<Float>.stride
        guard let inputBuffer = device.makeBuffer(
            bytes: input.storage,
            length: byteCount,
            options: .storageModeShared
        ) else {
            throw MetalTransposeError.bufferCreationFailed("input")
        }
        guard let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MetalTransposeError.bufferCreationFailed("output")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalTransposeError.commandCreationFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var dimensions = SIMD2<UInt32>(UInt32(rows), UInt32(columns))
        encoder.setBytes(&dimensions, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
        let groups = MTLSize(
            width: (columns + Self.tileWidth - 1) / Self.tileWidth,
            height: (rows + Self.tileWidth - 1) / Self.tileWidth,
            depth: 1
        )
        encoder.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: Self.tileWidth, height: Self.tileWidth, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        guard commandBuffer.status == .completed else {
            throw MetalTransposeError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "unknown error"
            )
        }
        let values = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().bindMemory(to: Float.self, capacity: input.storage.count),
                count: input.storage.count
            )
        )
        return try FloatTensor(values, shape: [columns, rows])
    }
}