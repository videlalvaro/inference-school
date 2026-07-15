import Foundation
import Metal

public enum MetalGEMVError: Error, LocalizedError {
    case noDevice
    case commandQueueCreationFailed
    case libraryCreationFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case unsupportedThreadgroupWidth(maximum: Int)
    case dimensionsTooLarge
    case bufferCreationFailed(String)
    case commandCreationFailed
    case commandFailed(String)
    case kernelResourceMissing(String)

    public var errorDescription: String? {
        switch self {
        case .noDevice: "Metal is unavailable on this machine."
        case .commandQueueCreationFailed: "Metal could not create a command queue."
        case let .libraryCreationFailed(message): "Metal could not compile the GEMV library: \(message)"
        case let .functionNotFound(name): "The Metal library does not contain \(name)."
        case let .pipelineCreationFailed(message): "Metal could not create the GEMV pipeline: \(message)"
        case let .unsupportedThreadgroupWidth(maximum): "GEMV requires 256 threads per group; the pipeline supports \(maximum)."
        case .dimensionsTooLarge: "GEMV dimensions exceed UInt32.max."
        case let .bufferCreationFailed(name): "Metal could not allocate the \(name) buffer."
        case .commandCreationFailed: "Metal could not create a command buffer or encoder."
        case let .commandFailed(message): "The Metal GEMV command failed: \(message)"
        case let .kernelResourceMissing(name): "The \(name) Metal source resource is missing."
        }
    }
}

public final class MetalGEMVPipeline {
    public static let threadgroupWidth = 256

    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "gemv_rows") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalGEMVError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalGEMVError.commandQueueCreationFailed
        }
        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalGEMVError.libraryCreationFailed(error.localizedDescription)
        }
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalGEMVError.functionNotFound(functionName)
        }
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalGEMVError.pipelineCreationFailed(error.localizedDescription)
        }
        guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else {
            throw MetalGEMVError.unsupportedThreadgroupWidth(
                maximum: pipeline.maxTotalThreadsPerThreadgroup
            )
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func multiply(matrix: FloatTensor, vector: FloatTensor) throws -> FloatTensor {
        guard matrix.rank == 2 else {
            throw TensorError.rankMismatch(expected: 2, actual: matrix.rank)
        }
        guard vector.rank == 1 else {
            throw TensorError.rankMismatch(expected: 1, actual: vector.rank)
        }
        let rows = matrix.shape[0]
        let columns = matrix.shape[1]
        guard columns == vector.shape[0] else {
            throw DenseLinearAlgebraError.innerDimensionMismatch(
                operation: "GEMV",
                lhs: columns,
                rhs: vector.shape[0]
            )
        }
        guard rows <= UInt32.max, columns <= UInt32.max else {
            throw MetalGEMVError.dimensionsTooLarge
        }
        guard rows > 0 else { return try FloatTensor([], shape: [0]) }
        guard columns > 0 else {
            return try FloatTensor(Array(repeating: 0, count: rows), shape: [rows])
        }

        let matrixBytes = matrix.storage.count * MemoryLayout<Float>.stride
        let vectorBytes = vector.storage.count * MemoryLayout<Float>.stride
        let outputBytes = rows * MemoryLayout<Float>.stride
        guard let matrixBuffer = device.makeBuffer(
            bytes: matrix.storage,
            length: matrixBytes,
            options: .storageModeShared
        ) else { throw MetalGEMVError.bufferCreationFailed("matrix") }
        guard let vectorBuffer = device.makeBuffer(
            bytes: vector.storage,
            length: vectorBytes,
            options: .storageModeShared
        ) else { throw MetalGEMVError.bufferCreationFailed("vector") }
        guard let outputBuffer = device.makeBuffer(length: outputBytes, options: .storageModeShared) else {
            throw MetalGEMVError.bufferCreationFailed("output")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalGEMVError.commandCreationFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        encoder.setBuffer(vectorBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var shape = SIMD2<UInt32>(UInt32(rows), UInt32(columns))
        encoder.setBytes(&shape, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw MetalGEMVError.commandFailed(commandBuffer.error?.localizedDescription ?? "unknown error")
        }

        let values = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float.self, capacity: rows),
            count: rows
        ))
        return try FloatTensor(values, shape: [rows])
    }
}