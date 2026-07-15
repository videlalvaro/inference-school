import Metal

public final class MetalTiledAttentionPipeline {
  public static let maximumHeadDimension = 128
  public static let keyTileSize = 16
  public static let threadgroupWidth = 128
  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState
  public init(source: String, functionName: String = "tiled_fused_attention") throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let queue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do { library = try device.makeLibrary(source: source, options: nil) } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: "tiled fused attention", message: error.localizedDescription)
    }
    guard let function = library.makeFunction(name: functionName) else {
      throw MetalNeuralOperatorError.functionNotFound(functionName)
    }
    do { pipeline = try device.makeComputePipelineState(function: function) } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "tiled fused attention", message: error.localizedDescription)
    }
    guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else {
      throw MetalNeuralOperatorError.unsupportedThreadgroupWidth(
        required: Self.threadgroupWidth, maximum: pipeline.maxTotalThreadsPerThreadgroup)
    }
    self.device = device
    commandQueue = queue
  }
  public func apply(
    _ q: FloatTensor, _ k: FloatTensor, _ v: FloatTensor, configuration c: AttentionConfiguration
  ) throws -> FloatTensor {
    let input = try AttentionInput(queries: q, keys: k, values: v, configuration: c)
    try validateVisibleKeys(input)
    guard c.headDimension <= Self.maximumHeadDimension else {
      throw AttentionError.unsupportedHeadDimension(
        maximum: Self.maximumHeadDimension, actual: c.headDimension)
    }
    guard q.elementCount > 0 else { return try FloatTensor([], shape: q.shape) }
    func buffer(_ values: [Float], _ name: String) throws -> any MTLBuffer {
      guard
        let b = device.makeBuffer(
          bytes: values, length: values.count * 4, options: .storageModeShared)
      else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return b
    }
    let qb = try buffer(q.storage, "queries")
    let kb = try buffer(k.storage, "keys")
    let vb = try buffer(v.storage, "values")
    guard let out = device.makeBuffer(length: q.elementCount * 4, options: .storageModeShared),
      let command = commandQueue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw MetalNeuralOperatorError.commandCreationFailed }
    var shape = SIMD4<UInt32>(
      UInt32(input.queryLength), UInt32(input.keyValueLength), UInt32(c.queryHeadCount),
      UInt32(c.keyValueHeadCount))
    var dims = SIMD4<UInt32>(
      UInt32(c.headDimension), UInt32(c.groupSize), UInt32(c.queryPositionOffset),
      UInt32(c.keyPositionOffset))
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(qb, offset: 0, index: 0)
    encoder.setBuffer(kb, offset: 0, index: 1)
    encoder.setBuffer(vb, offset: 0, index: 2)
    encoder.setBuffer(out, offset: 0, index: 3)
    encoder.setBytes(&shape, length: 16, index: 4)
    encoder.setBytes(&dims, length: 16, index: 5)
    encoder.dispatchThreadgroups(
      MTLSize(width: input.queryLength * c.queryHeadCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: "tiled fused attention",
        message: command.error?.localizedDescription ?? "unknown error")
    }
    let values = Array(
      UnsafeBufferPointer(
        start: out.contents().bindMemory(to: Float.self, capacity: q.elementCount),
        count: q.elementCount))
    return try FloatTensor(values, shape: q.shape)
  }
}
