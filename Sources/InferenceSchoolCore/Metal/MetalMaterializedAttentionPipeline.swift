import Metal

public final class MetalMaterializedAttentionPipeline {
  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let scoresPipeline: any MTLComputePipelineState
  private let applyPipeline: any MTLComputePipelineState

  public init(
    source: String,
    scoresFunctionName: String = "causal_attention_scores",
    applyFunctionName: String = "causal_attention_apply"
  ) throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do { library = try device.makeLibrary(source: source, options: nil) } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: "materialized attention", message: error.localizedDescription)
    }
    guard let scoresFunction = library.makeFunction(name: scoresFunctionName) else {
      throw MetalNeuralOperatorError.functionNotFound(scoresFunctionName)
    }
    guard let applyFunction = library.makeFunction(name: applyFunctionName) else {
      throw MetalNeuralOperatorError.functionNotFound(applyFunctionName)
    }
    do {
      scoresPipeline = try device.makeComputePipelineState(function: scoresFunction)
      applyPipeline = try device.makeComputePipelineState(function: applyFunction)
    } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "materialized attention", message: error.localizedDescription)
    }
    self.device = device
    self.commandQueue = commandQueue
  }

  public func apply(
    _ queries: FloatTensor,
    _ keys: FloatTensor,
    _ values: FloatTensor,
    configuration: AttentionConfiguration
  ) throws -> FloatTensor {
    let input = try P016CausalAttentionContract.validate(
      queries: queries, keys: keys, values: values, configuration: configuration)
    guard
      [
        input.queryLength, input.keyValueLength, configuration.headDimension,
        configuration.queryPositionOffset, configuration.keyPositionOffset,
      ].allSatisfy({ $0 <= UInt32.max })
    else {
      throw MetalNeuralOperatorError.dimensionsTooLarge
    }
    guard input.queryLength > 0 else { return try FloatTensor([], shape: queries.shape) }
    let queryBuffer = try makeBuffer(queries.storage, name: "attention queries")
    let keyBuffer = try makeBuffer(keys.storage, name: "attention keys")
    let valueBuffer = try makeBuffer(values.storage, name: "attention values")
    let scoreCount = input.queryLength * input.keyValueLength
    guard
      let scoresBuffer = device.makeBuffer(
        length: scoreCount * MemoryLayout<Float>.stride, options: .storageModeShared),
      let outputBuffer = device.makeBuffer(
        length: queries.elementCount * MemoryLayout<Float>.stride, options: .storageModeShared)
    else {
      throw MetalNeuralOperatorError.bufferCreationFailed("materialized attention output")
    }
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      throw MetalNeuralOperatorError.commandCreationFailed
    }
    var shape = SIMD4<UInt32>(
      UInt32(input.queryLength), UInt32(input.keyValueLength), UInt32(configuration.headDimension),
      0)
    var offsets = SIMD2<UInt32>(
      UInt32(configuration.queryPositionOffset), UInt32(configuration.keyPositionOffset))

    guard let scoresEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw MetalNeuralOperatorError.commandCreationFailed
    }
    scoresEncoder.setComputePipelineState(scoresPipeline)
    scoresEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
    scoresEncoder.setBuffer(keyBuffer, offset: 0, index: 1)
    scoresEncoder.setBuffer(scoresBuffer, offset: 0, index: 2)
    scoresEncoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
    scoresEncoder.setBytes(&offsets, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
    scoresEncoder.dispatchThreads(
      MTLSize(width: scoreCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(256, scoresPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    )
    scoresEncoder.endEncoding()

    guard let applyEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw MetalNeuralOperatorError.commandCreationFailed
    }
    applyEncoder.setComputePipelineState(applyPipeline)
    applyEncoder.setBuffer(scoresBuffer, offset: 0, index: 0)
    applyEncoder.setBuffer(valueBuffer, offset: 0, index: 1)
    applyEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
    applyEncoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
    applyEncoder.dispatchThreads(
      MTLSize(width: input.queryLength, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(256, applyPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    )
    applyEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: "materialized attention",
        message: commandBuffer.error?.localizedDescription ?? "unknown error")
    }
    let output = Array(
      UnsafeBufferPointer(
        start: outputBuffer.contents().bindMemory(to: Float.self, capacity: queries.elementCount),
        count: queries.elementCount))
    return try FloatTensor(output, shape: queries.shape)
  }

  private func makeBuffer(_ values: [Float], name: String) throws -> any MTLBuffer {
    guard
      let buffer = device.makeBuffer(
        bytes: values, length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared)
    else {
      throw MetalNeuralOperatorError.bufferCreationFailed(name)
    }
    return buffer
  }
}
