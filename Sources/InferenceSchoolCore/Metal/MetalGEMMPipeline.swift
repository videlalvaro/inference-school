import Foundation
import Metal

public enum MetalGEMMError: Error, LocalizedError {
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
        case let .libraryCreationFailed(message): "Metal could not compile the GEMM library: \(message)"
        case let .functionNotFound(name): "The Metal library does not contain \(name)."
        case let .pipelineCreationFailed(message): "Metal could not create the GEMM pipeline: \(message)"
        case .dimensionsTooLarge: "GEMM dimensions exceed UInt32.max."
        case let .bufferCreationFailed(name): "Metal could not allocate the \(name) buffer."
        case .commandCreationFailed: "Metal could not create a command buffer or encoder."
        case let .commandFailed(message): "The Metal GEMM command failed: \(message)"
        case let .kernelResourceMissing(name): "The \(name) Metal source resource is missing."
        }
    }
}

public final class MetalGEMMPipeline {
    public static let tileWidth = 16

    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "tiled_gemm") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalGEMMError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalGEMMError.commandQueueCreationFailed
        }
        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalGEMMError.libraryCreationFailed(error.localizedDescription)
        }
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalGEMMError.functionNotFound(functionName)
        }
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalGEMMError.pipelineCreationFailed(error.localizedDescription)
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func multiply(_ lhs: FloatTensor, _ rhs: FloatTensor) throws -> FloatTensor {
        guard lhs.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: lhs.rank)
        }
        guard rhs.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: rhs.rank)
        }
        let m = lhs.shape[0]
        let k = lhs.shape[1]
        let n = rhs.shape[1]
        guard k == rhs.shape[0] else {
            throw DenseLinearAlgebraError.innerDimensionMismatch(
                operation: "GEMM",
                lhs: k,
                rhs: rhs.shape[0]
            )
        }
        guard m <= UInt32.max, k <= UInt32.max, n <= UInt32.max else {
            throw MetalGEMMError.dimensionsTooLarge
        }
        guard m > 0, n > 0 else { return try FloatTensor([], shape: [m, n]) }
        guard k > 0 else {
            return try FloatTensor(Array(repeating: 0, count: m * n), shape: [m, n])
        }

        let lhsBytes = lhs.storage.count * MemoryLayout<Float>.stride
        let rhsBytes = rhs.storage.count * MemoryLayout<Float>.stride
        let outputCount = m * n
        guard let lhsBuffer = device.makeBuffer(
            bytes: lhs.storage,
            length: lhsBytes,
            options: .storageModeShared
        ) else { throw MetalGEMMError.bufferCreationFailed("left matrix") }
        guard let rhsBuffer = device.makeBuffer(
            bytes: rhs.storage,
            length: rhsBytes,
            options: .storageModeShared
        ) else { throw MetalGEMMError.bufferCreationFailed("right matrix") }
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { throw MetalGEMMError.bufferCreationFailed("output") }
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalGEMMError.commandCreationFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
        encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var shape = SIMD4<UInt32>(UInt32(m), UInt32(k), UInt32(n), 0)
        encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (n + Self.tileWidth - 1) / Self.tileWidth,
                height: (m + Self.tileWidth - 1) / Self.tileWidth,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: Self.tileWidth, height: Self.tileWidth, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw MetalGEMMError.commandFailed(commandBuffer.error?.localizedDescription ?? "unknown error")
        }

        let values = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount),
            count: outputCount
        ))
        return try FloatTensor(values, shape: [m, n])
    }
}