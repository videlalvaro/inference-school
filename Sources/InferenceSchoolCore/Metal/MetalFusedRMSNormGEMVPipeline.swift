import Metal

public final class MetalFusedRMSNormGEMVPipeline {
    public static let threadgroupWidth = 256

    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "fused_rmsnorm_gemv") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalNeuralOperatorError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalNeuralOperatorError.commandQueueCreationFailed }
        let library: any MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw MetalNeuralOperatorError.libraryCreationFailed(operation: "fused RMSNorm+GEMV", message: error.localizedDescription) }
        guard let function = library.makeFunction(name: functionName) else { throw MetalNeuralOperatorError.functionNotFound(functionName) }
        do { pipeline = try device.makeComputePipelineState(function: function) }
        catch { throw MetalNeuralOperatorError.pipelineCreationFailed(operation: "fused RMSNorm+GEMV", message: error.localizedDescription) }
        guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else { throw MetalNeuralOperatorError.unsupportedThreadgroupWidth(required: Self.threadgroupWidth, maximum: pipeline.maxTotalThreadsPerThreadgroup) }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func project(_ input: FloatTensor, gamma: FloatTensor, weights: FloatTensor, epsilon: Float) throws -> FloatTensor {
        try validate(input, gamma: gamma, weights: weights, epsilon: epsilon)
        let outputs = weights.shape[0]
        let width = input.shape[0]
        guard outputs > 0 else { return try FloatTensor([], shape: [0]) }
        guard let inputBuffer = device.makeBuffer(bytes: input.storage, length: width * MemoryLayout<Float>.stride, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("fused input") }
        guard let gammaBuffer = device.makeBuffer(bytes: gamma.storage, length: width * MemoryLayout<Float>.stride, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("fused gamma") }
        guard let weightsBuffer = device.makeBuffer(bytes: weights.storage, length: weights.elementCount * MemoryLayout<Float>.stride, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("projection weights") }
        guard let outputBuffer = device.makeBuffer(length: outputs * MemoryLayout<Float>.stride, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("projection output") }
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let encoder = commandBuffer.makeComputeCommandEncoder() else { throw MetalNeuralOperatorError.commandCreationFailed }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(gammaBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        var shape = SIMD2<UInt32>(UInt32(outputs), UInt32(width))
        var epsilonValue = epsilon
        encoder.setBytes(&shape, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.stride, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: outputs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else { throw MetalNeuralOperatorError.commandFailed(operation: "fused RMSNorm+GEMV", message: commandBuffer.error?.localizedDescription ?? "unknown error") }
        let values = Array(UnsafeBufferPointer(start: outputBuffer.contents().bindMemory(to: Float.self, capacity: outputs), count: outputs))
        return try FloatTensor(values, shape: [outputs])
    }

    private func validate(_ input: FloatTensor, gamma: FloatTensor, weights: FloatTensor, epsilon: Float) throws {
        guard input.rank == 1 else { throw TensorError.rankMismatch(expected: 1, actual: input.rank) }
        guard gamma.rank == 1 else { throw TensorError.rankMismatch(expected: 1, actual: gamma.rank) }
        guard weights.rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: weights.rank) }
        guard !input.storage.isEmpty else { throw FusedRMSNormGEMVError.emptyInput }
        guard weights.shape[1] == input.shape[0] else { throw FusedRMSNormGEMVError.inputWidthMismatch(expected: weights.shape[1], actual: input.shape[0]) }
        guard gamma.shape[0] == input.shape[0] else { throw FusedRMSNormGEMVError.gammaWidthMismatch(expected: input.shape[0], actual: gamma.shape[0]) }
        guard epsilon.isFinite, epsilon > 0 else { throw FusedRMSNormGEMVError.invalidEpsilon(epsilon) }
        guard weights.shape[0] <= UInt32.max, input.shape[0] <= UInt32.max else { throw MetalNeuralOperatorError.dimensionsTooLarge }
    }
}