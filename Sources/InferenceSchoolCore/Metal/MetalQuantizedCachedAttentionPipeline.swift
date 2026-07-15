import Metal

public final class MetalQuantizedCachedAttentionPipeline {
  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState

  public init(source: String, functionName: String = "quantized_cached_decode_attention") throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do { library = try device.makeLibrary(source: source, options: nil) } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: "quantized cached decode attention", message: error.localizedDescription)
    }
    guard let function = library.makeFunction(name: functionName) else {
      throw MetalNeuralOperatorError.functionNotFound(functionName)
    }
    do { pipeline = try device.makeComputePipelineState(function: function) } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "quantized cached decode attention", message: error.localizedDescription)
    }
    self.device = device
    self.commandQueue = commandQueue
  }

  public func apply(
    query: FloatTensor,
    keyStorage: [Int8],
    valueStorage: [Int8],
    keyScales: [Float],
    valueScales: [Float],
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
    let vectorCount = configuration.layerCount * configuration.capacity
      * configuration.keyValueHeadCount
    guard keyStorage.count == configuration.elementsPerTensor,
      valueStorage.count == configuration.elementsPerTensor,
      keyScales.count == vectorCount,
      valueScales.count == vectorCount
    else { throw TensorError.storageCountMismatch(expected: configuration.elementsPerTensor, actual: keyStorage.count) }
    guard tokenCount > 0 else { throw KVCacheError.invalidTokenCount(tokenCount) }

    func floatBuffer(_ values: [Float], name: String) throws -> any MTLBuffer {
      guard let result = device.makeBuffer(
        bytes: values,
        length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared)
      else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return result
    }
    func int8Buffer(_ values: [Int8], name: String) throws -> any MTLBuffer {
      let result = values.withUnsafeBytes { bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      }
      guard let result else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return result
    }
    let queryBuffer = try floatBuffer(query.storage, name: "query")
    let keyBuffer = try int8Buffer(keyStorage, name: "quantized keys")
    let valueBuffer = try int8Buffer(valueStorage, name: "quantized values")
    let keyScaleBuffer = try floatBuffer(keyScales, name: "key scales")
    let valueScaleBuffer = try floatBuffer(valueScales, name: "value scales")
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
    encoder.setBuffer(keyScaleBuffer, offset: 0, index: 3)
    encoder.setBuffer(valueScaleBuffer, offset: 0, index: 4)
    encoder.setBuffer(output, offset: 0, index: 5)
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 6)
    encoder.setBytes(&dimensions, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 7)
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
        operation: "quantized cached decode attention",
        message: command.error?.localizedDescription ?? "unknown error")
    }
    let values = Array(UnsafeBufferPointer(
      start: output.contents().bindMemory(to: Float.self, capacity: query.elementCount),
      count: query.elementCount))
    return try FloatTensor(values, shape: query.shape)
  }
}
