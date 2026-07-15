import Metal

public final class MetalRMSNormPipeline {
    public static let threadgroupWidth = 256

    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "rmsnorm_rows") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalNeuralOperatorError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalNeuralOperatorError.commandQueueCreationFailed }
        let library: any MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw MetalNeuralOperatorError.libraryCreationFailed(operation: "RMSNorm", message: error.localizedDescription) }
        guard let function = library.makeFunction(name: functionName) else { throw MetalNeuralOperatorError.functionNotFound(functionName) }
        do { pipeline = try device.makeComputePipelineState(function: function) }
        catch { throw MetalNeuralOperatorError.pipelineCreationFailed(operation: "RMSNorm", message: error.localizedDescription) }
        guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else { throw MetalNeuralOperatorError.unsupportedThreadgroupWidth(required: Self.threadgroupWidth, maximum: pipeline.maxTotalThreadsPerThreadgroup) }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func apply(_ input: FloatTensor, gamma: FloatTensor, epsilon: Float) throws -> FloatTensor {
        try validate(input, gamma: gamma, epsilon: epsilon)
        let rows = input.shape[0]
        let width = input.shape[1]
        guard rows > 0 else { return input }
        let inputBytes = input.elementCount * MemoryLayout<Float>.stride
        let gammaBytes = width * MemoryLayout<Float>.stride
        guard let inputBuffer = device.makeBuffer(bytes: input.storage, length: inputBytes, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("RMSNorm input") }
        guard let gammaBuffer = device.makeBuffer(bytes: gamma.storage, length: gammaBytes, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("RMSNorm gamma") }
        guard let outputBuffer = device.makeBuffer(length: inputBytes, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("RMSNorm output") }
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let encoder = commandBuffer.makeComputeCommandEncoder() else { throw MetalNeuralOperatorError.commandCreationFailed }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(gammaBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var shape = SIMD2<UInt32>(UInt32(rows), UInt32(width))
        var epsilonValue = epsilon
        encoder.setBytes(&shape, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.stride, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else { throw MetalNeuralOperatorError.commandFailed(operation: "RMSNorm", message: commandBuffer.error?.localizedDescription ?? "unknown error") }
        let values = Array(UnsafeBufferPointer(start: outputBuffer.contents().bindMemory(to: Float.self, capacity: input.elementCount), count: input.elementCount))
        return try FloatTensor(values, shape: input.shape)
    }

    private func validate(_ input: FloatTensor, gamma: FloatTensor, epsilon: Float) throws {
        guard input.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: input.rank) }
        guard gamma.rank == 1 else { throw TensorError.rankMismatch(expected: 1, actual: gamma.rank) }
        guard input.shape[1] > 0 else { throw RMSNormError.emptyFeatureWidth }
        guard gamma.shape[0] == input.shape[1] else { throw RMSNormError.gammaWidthMismatch(expected: input.shape[1], actual: gamma.shape[0]) }
        guard epsilon.isFinite, epsilon > 0 else { throw RMSNormError.invalidEpsilon(epsilon) }
        guard input.shape[0] <= UInt32.max, input.shape[1] <= UInt32.max else { throw MetalNeuralOperatorError.dimensionsTooLarge }
        for row in 0..<input.shape[0] {
            for column in 0..<input.shape[1] where !input.storage[row * input.shape[1] + column].isFinite { throw RMSNormError.nonFiniteInput(row: row, column: column) }
        }
    }
}