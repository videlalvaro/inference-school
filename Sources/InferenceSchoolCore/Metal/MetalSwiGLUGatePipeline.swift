import Metal

public final class MetalSwiGLUGatePipeline {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    public init(source: String, functionName: String = "swiglu_gate") throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalNeuralOperatorError.noDevice }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalNeuralOperatorError.commandQueueCreationFailed }
        let library: any MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw MetalNeuralOperatorError.libraryCreationFailed(operation: "SwiGLU gate", message: error.localizedDescription) }
        guard let function = library.makeFunction(name: functionName) else { throw MetalNeuralOperatorError.functionNotFound(functionName) }
        do { pipeline = try device.makeComputePipelineState(function: function) }
        catch { throw MetalNeuralOperatorError.pipelineCreationFailed(operation: "SwiGLU gate", message: error.localizedDescription) }
        self.device = device
        self.commandQueue = commandQueue
    }

    public func apply(_ gate: FloatTensor, _ up: FloatTensor) throws -> FloatTensor {
        guard gate.shape == up.shape else { throw SwiGLUError.gateValueShapeMismatch(gate: gate.shape, up: up.shape) }
        guard gate.elementCount <= UInt32.max else { throw MetalNeuralOperatorError.dimensionsTooLarge }
        guard gate.elementCount > 0 else { return gate }
        let byteCount = gate.elementCount * MemoryLayout<Float>.stride
        guard let gateBuffer = device.makeBuffer(bytes: gate.storage, length: byteCount, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("gate") }
        guard let upBuffer = device.makeBuffer(bytes: up.storage, length: byteCount, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("up") }
        guard let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else { throw MetalNeuralOperatorError.bufferCreationFailed("gated output") }
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let encoder = commandBuffer.makeComputeCommandEncoder() else { throw MetalNeuralOperatorError.commandCreationFailed }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var count = UInt32(gate.elementCount)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: gate.elementCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else { throw MetalNeuralOperatorError.commandFailed(operation: "SwiGLU gate", message: commandBuffer.error?.localizedDescription ?? "unknown error") }
        let values = Array(UnsafeBufferPointer(start: outputBuffer.contents().bindMemory(to: Float.self, capacity: gate.elementCount), count: gate.elementCount))
        return try FloatTensor(values, shape: gate.shape)
    }
}