import Metal

public final class MetalActivationPipeline {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "activation_elementwise") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalNeuralOperatorError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalNeuralOperatorError.commandQueueCreationFailed }
        let library: any MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw MetalNeuralOperatorError.libraryCreationFailed(operation: "activation", message: error.localizedDescription) }
        guard let function = library.makeFunction(name: functionName) else { throw MetalNeuralOperatorError.functionNotFound(functionName) }
        do { pipeline = try device.makeComputePipelineState(function: function) }
        catch { throw MetalNeuralOperatorError.pipelineCreationFailed(operation: "activation", message: error.localizedDescription) }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func apply(_ input: FloatTensor, activation: Activation) throws -> FloatTensor {
        guard input.elementCount <= UInt32.max else { throw MetalNeuralOperatorError.dimensionsTooLarge }
        guard input.elementCount > 0 else { return input }
        let byteCount = input.elementCount * MemoryLayout<Float>.stride
        guard let inputBuffer = device.makeBuffer(bytes: input.storage, length: byteCount, options: .storageModeShared) else {
            throw MetalNeuralOperatorError.bufferCreationFailed("activation input")
        }
        guard let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MetalNeuralOperatorError.bufferCreationFailed("activation output")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalNeuralOperatorError.commandCreationFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var count = UInt32(input.elementCount)
        var kind = activation.rawValue
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&kind, length: MemoryLayout<UInt32>.stride, index: 3)
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: input.elementCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw MetalNeuralOperatorError.commandFailed(operation: "activation", message: commandBuffer.error?.localizedDescription ?? "unknown error")
        }
        let values = Array(UnsafeBufferPointer(start: outputBuffer.contents().bindMemory(to: Float.self, capacity: input.elementCount), count: input.elementCount))
        return try FloatTensor(values, shape: input.shape)
    }
}