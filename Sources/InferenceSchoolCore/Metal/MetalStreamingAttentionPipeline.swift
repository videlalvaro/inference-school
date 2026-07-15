import Metal

public final class MetalStreamingAttentionPipeline {
  public static let maximumHeadDimension = 128
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
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, configuration c: AttentionConfiguration,
    window: Int? = nil
  ) throws -> FloatTensor {
    let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
    try validateVisibleKeys(input, window: window)
    guard c.headDimension <= Self.maximumHeadDimension else {
      throw AttentionError.unsupportedHeadDimension(
        maximum: Self.maximumHeadDimension, actual: c.headDimension)
    }
    guard
      [
        input.queryLength, input.keyValueLength, c.queryHeadCount, c.keyValueHeadCount,
        c.headDimension, c.queryPositionOffset, c.keyPositionOffset, window ?? 0,
      ].allSatisfy({ $0 <= UInt32.max })
    else { throw MetalNeuralOperatorError.dimensionsTooLarge }
    guard q.elementCount > 0 else { return try FloatTensor([], shape: q.shape) }
    func buffer(_ values: [Float], _ name: String) throws -> any MTLBuffer {
      guard
        let result = device.makeBuffer(
          bytes: values, length: values.count * MemoryLayout<Float>.stride,
          options: .storageModeShared)
      else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return result
    }
    let qb = try buffer(q.storage, "queries")
    let kb = try buffer(k.storage, "keys")
    let vb = try buffer(v.storage, "values")
    guard
      let output = device.makeBuffer(
        length: q.elementCount * MemoryLayout<Float>.stride, options: .storageModeShared),
      let command = commandQueue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw MetalNeuralOperatorError.commandCreationFailed }
    var shape = SIMD4<UInt32>(
      UInt32(input.queryLength), UInt32(input.keyValueLength), UInt32(c.queryHeadCount),
      UInt32(c.keyValueHeadCount))
    var dimensions = SIMD4<UInt32>(
      UInt32(c.headDimension), UInt32(c.groupSize), UInt32(c.queryPositionOffset),
      UInt32(c.keyPositionOffset))
    var metalWindow = UInt32(window ?? 0)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(qb, offset: 0, index: 0)
    encoder.setBuffer(kb, offset: 0, index: 1)
    encoder.setBuffer(vb, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 4)
    encoder.setBytes(&dimensions, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 5)
    encoder.setBytes(&metalWindow, length: MemoryLayout<UInt32>.stride, index: 6)
    encoder.dispatchThreads(
      MTLSize(width: input.queryLength * c.queryHeadCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: operation, message: command.error?.localizedDescription ?? "unknown error")
    }
    let values = Array(
      UnsafeBufferPointer(
        start: output.contents().bindMemory(to: Float.self, capacity: q.elementCount),
        count: q.elementCount))
    return try FloatTensor(values, shape: q.shape)
  }
}
