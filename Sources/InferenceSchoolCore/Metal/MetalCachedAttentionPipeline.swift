import Metal

public final class MetalCachedAttentionPipeline {
  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState
  private let operation: String

  public init(source: String, functionName: String = "cached_decode_attention") throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do { library = try device.makeLibrary(source: source, options: nil) } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: "cached decode attention", message: error.localizedDescription)
    }
    guard let function = library.makeFunction(name: functionName) else {
      throw MetalNeuralOperatorError.functionNotFound(functionName)
    }
    do { pipeline = try device.makeComputePipelineState(function: function) } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "cached decode attention", message: error.localizedDescription)
    }
    self.device = device
    self.commandQueue = commandQueue
    self.operation = "cached decode attention"
  }

  public func apply(
    query: FloatTensor,
    keyStorage: [Float],
    valueStorage: [Float],
    configuration: KVCacheConfiguration,
    queryHeadCount: Int,
    layer: Int,
    tokenCount: Int
  ) throws -> FloatTensor {
    try configuration.validate(layer: layer)
    let attention = try AttentionConfiguration(
      queryHeadCount: queryHeadCount,
      keyValueHeadCount: configuration.keyValueHeadCount,
      headDimension: configuration.headDimension)
    let expectedQuery = [queryHeadCount, configuration.headDimension]
    guard query.shape == expectedQuery else {
      throw KVCacheError.vectorShapeMismatch(
        name: "Query", expected: expectedQuery, actual: query.shape)
    }
    guard keyStorage.count == configuration.elementsPerTensor else {
      throw TensorError.storageCountMismatch(
        expected: configuration.elementsPerTensor, actual: keyStorage.count)
    }
    guard valueStorage.count == configuration.elementsPerTensor else {
      throw TensorError.storageCountMismatch(
        expected: configuration.elementsPerTensor, actual: valueStorage.count)
    }
    guard tokenCount > 0 else { throw KVCacheError.invalidTokenCount(tokenCount) }
    guard tokenCount <= configuration.capacity else {
      throw KVCacheError.capacityExceeded(layer: layer, capacity: configuration.capacity)
    }
    guard [queryHeadCount, configuration.keyValueHeadCount, configuration.headDimension,
      configuration.capacity, layer, tokenCount].allSatisfy({ $0 <= UInt32.max })
    else { throw MetalNeuralOperatorError.dimensionsTooLarge }

    func buffer(_ values: [Float], name: String) throws -> any MTLBuffer {
      guard let result = device.makeBuffer(
        bytes: values,
        length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared)
      else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return result
    }
    let queryBuffer = try buffer(query.storage, name: "query")
    let keyBuffer = try buffer(keyStorage, name: "key cache")
    let valueBuffer = try buffer(valueStorage, name: "value cache")
    guard let output = device.makeBuffer(
      length: query.elementCount * MemoryLayout<Float>.stride, options: .storageModeShared),
      let command = commandQueue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw MetalNeuralOperatorError.commandCreationFailed }
    var shape = SIMD4<UInt32>(
      UInt32(layer), UInt32(configuration.capacity), UInt32(tokenCount), UInt32(queryHeadCount))
    var dimensions = SIMD4<UInt32>(
      UInt32(configuration.keyValueHeadCount), UInt32(configuration.headDimension),
      UInt32(attention.groupSize), 0)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(queryBuffer, offset: 0, index: 0)
    encoder.setBuffer(keyBuffer, offset: 0, index: 1)
    encoder.setBuffer(valueBuffer, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 4)
    encoder.setBytes(&dimensions, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 5)
    encoder.dispatchThreads(
      MTLSize(width: configuration.headDimension, height: queryHeadCount, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(configuration.headDimension, pipeline.maxTotalThreadsPerThreadgroup),
        height: 1,
        depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: operation, message: command.error?.localizedDescription ?? "unknown error")
    }
    let values = Array(UnsafeBufferPointer(
      start: output.contents().bindMemory(to: Float.self, capacity: query.elementCount),
      count: query.elementCount))
    return try FloatTensor(values, shape: query.shape)
  }
}
