import Metal

public final class MetalSoftmaxPipeline {
    public static let threadgroupWidth = 256
    public static let maximumRowWidth = 1024

    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "softmax_rows") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalNeuralOperatorError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalNeuralOperatorError.commandQueueCreationFailed }
        let library: any MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw MetalNeuralOperatorError.libraryCreationFailed(operation: "softmax", message: error.localizedDescription) }
        guard let function = library.makeFunction(name: functionName) else { throw MetalNeuralOperatorError.functionNotFound(functionName) }
        do { pipeline = try device.makeComputePipelineState(function: function) }
        catch { throw MetalNeuralOperatorError.pipelineCreationFailed(operation: "softmax", message: error.localizedDescription) }
        guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else {
            throw MetalNeuralOperatorError.unsupportedThreadgroupWidth(required: Self.threadgroupWidth, maximum: pipeline.maxTotalThreadsPerThreadgroup)
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func apply(_ logits: FloatTensor) throws -> FloatTensor {
        try validate(logits)
        let rows = logits.shape[0]
        let columns = logits.shape[1]
        guard rows > 0 else { return logits }
        let byteCount = logits.elementCount * MemoryLayout<Float>.stride
        guard let inputBuffer = device.makeBuffer(bytes: logits.storage, length: byteCount, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("softmax input") }
        guard let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("softmax output") }
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let encoder = commandBuffer.makeComputeCommandEncoder() else { throw MetalNeuralOperatorError.commandCreationFailed }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var shape = SIMD2<UInt32>(UInt32(rows), UInt32(columns))
        encoder.setBytes(&shape, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else { throw MetalNeuralOperatorError.commandFailed(operation: "softmax", message: commandBuffer.error?.localizedDescription ?? "unknown error") }
        let values = Array(UnsafeBufferPointer(start: outputBuffer.contents().bindMemory(to: Float.self, capacity: logits.elementCount), count: logits.elementCount))
        return try FloatTensor(values, shape: logits.shape)
    }

    private func validate(_ logits: FloatTensor) throws {
        guard logits.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: logits.rank) }
        guard logits.shape[1] > 0 else { throw SoftmaxError.emptyRow }
        guard logits.shape[1] <= Self.maximumRowWidth else { throw MetalNeuralOperatorError.rowWidthExceedsMaximum(maximum: Self.maximumRowWidth, actual: logits.shape[1]) }
        guard logits.shape[0] <= UInt32.max else { throw MetalNeuralOperatorError.dimensionsTooLarge }
        for row in 0..<logits.shape[0] {
            for column in 0..<logits.shape[1] where !logits.storage[row * logits.shape[1] + column].isFinite {
                throw SoftmaxError.nonFiniteInput(row: row, column: column)
            }
        }
    }
}