import Metal

public final class MetalParallelAttentionPipeline {
  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState
  private let operation: String

  public init(source: String, functionName: String, operation: String) throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do { library = try device.makeLibrary(source: source, options: nil) } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: operation, message: error.localizedDescription)
    }
    guard let function = library.makeFunction(name: functionName) else {
      throw MetalNeuralOperatorError.functionNotFound(functionName)
    }
    do { pipeline = try device.makeComputePipelineState(function: function) } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: operation, message: error.localizedDescription)
    }
    self.device = device
    self.commandQueue = commandQueue
    self.operation = operation
  }

  public func apply(
    _ queries: FloatTensor, _ keys: FloatTensor, _ values: FloatTensor,
    configuration: AttentionConfiguration
  ) throws -> FloatTensor {
    let input = try AttentionInput(
      queries: queries, keys: keys, values: values, configuration: configuration)
    try validateVisibleKeys(input)
    guard
      [
        input.queryLength, input.keyValueLength, configuration.queryHeadCount,
        configuration.keyValueHeadCount, configuration.headDimension,
        configuration.queryPositionOffset, configuration.keyPositionOffset,
      ].allSatisfy({ $0 <= UInt32.max })
    else { throw MetalNeuralOperatorError.dimensionsTooLarge }
    guard queries.elementCount > 0 else { return try FloatTensor([], shape: queries.shape) }
    func buffer(_ values: [Float], _ name: String) throws -> any MTLBuffer {
      guard
        let result = device.makeBuffer(
          bytes: values, length: values.count * MemoryLayout<Float>.stride,
          options: .storageModeShared)
      else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return result
    }
    let queryBuffer = try buffer(queries.storage, "queries")
    let keyBuffer = try buffer(keys.storage, "keys")
    let valueBuffer = try buffer(values.storage, "values")
    guard
      let outputBuffer = device.makeBuffer(
        length: queries.elementCount * MemoryLayout<Float>.stride, options: .storageModeShared),
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else { throw MetalNeuralOperatorError.commandCreationFailed }
    var shape = SIMD4<UInt32>(
      UInt32(input.queryLength), UInt32(input.keyValueLength), UInt32(configuration.queryHeadCount),
      UInt32(configuration.keyValueHeadCount))
    var dimensions = SIMD4<UInt32>(
      UInt32(configuration.headDimension), UInt32(configuration.groupSize),
      UInt32(configuration.queryPositionOffset), UInt32(configuration.keyPositionOffset))
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(queryBuffer, offset: 0, index: 0)
    encoder.setBuffer(keyBuffer, offset: 0, index: 1)
    encoder.setBuffer(valueBuffer, offset: 0, index: 2)
    encoder.setBuffer(outputBuffer, offset: 0, index: 3)
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 4)
    encoder.setBytes(&dimensions, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 5)
    encoder.dispatchThreads(
      MTLSize(width: input.queryLength * configuration.queryHeadCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: operation, message: commandBuffer.error?.localizedDescription ?? "unknown error")
    }
    let output = Array(
      UnsafeBufferPointer(
        start: outputBuffer.contents().bindMemory(to: Float.self, capacity: queries.elementCount),
        count: queries.elementCount))
    return try FloatTensor(output, shape: queries.shape)
  }
}
