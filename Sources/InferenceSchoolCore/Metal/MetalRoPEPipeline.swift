import Metal

public final class MetalRoPEPipeline {
  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState

  public init(source: String, functionName: String = "rope_adjacent_pairs") throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do { library = try device.makeLibrary(source: source, options: nil) } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: "RoPE", message: error.localizedDescription)
    }
    guard let function = library.makeFunction(name: functionName) else {
      throw MetalNeuralOperatorError.functionNotFound(functionName)
    }
    do { pipeline = try device.makeComputePipelineState(function: function) } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "RoPE", message: error.localizedDescription)
    }
    self.device = device
    self.commandQueue = commandQueue
  }

  public func apply(
    _ queries: FloatTensor,
    _ keys: FloatTensor,
    rotaryDimension: Int,
    base: Float,
    queryPositionOffset: Int,
    keyPositionOffset: Int
  ) throws -> RoPEResult {
    try RoPEContract.validate(
      queries: queries,
      keys: keys,
      rotaryDimension: rotaryDimension,
      base: base,
      queryPositionOffset: queryPositionOffset,
      keyPositionOffset: keyPositionOffset
    )
    return RoPEResult(
      queries: try rotate(
        queries, rotaryDimension: rotaryDimension, base: base, positionOffset: queryPositionOffset),
      keys: try rotate(
        keys, rotaryDimension: rotaryDimension, base: base, positionOffset: keyPositionOffset)
    )
  }

  private func rotate(
    _ tensor: FloatTensor,
    rotaryDimension: Int,
    base: Float,
    positionOffset: Int
  ) throws -> FloatTensor {
    let pairCount = rotaryDimension / 2
    let workItemCount = tensor.shape[0] * tensor.shape[1] * pairCount
    guard tensor.shape.allSatisfy({ $0 <= UInt32.max }), positionOffset <= UInt32.max else {
      throw MetalNeuralOperatorError.dimensionsTooLarge
    }
    guard workItemCount > 0 else { return tensor }
    let byteCount = tensor.elementCount * MemoryLayout<Float>.stride
    guard
      let input = device.makeBuffer(
        bytes: tensor.storage, length: byteCount, options: .storageModeShared),
      let output = device.makeBuffer(
        bytes: tensor.storage, length: byteCount, options: .storageModeShared)
    else {
      throw MetalNeuralOperatorError.bufferCreationFailed("RoPE input/output")
    }
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      throw MetalNeuralOperatorError.commandCreationFailed
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    var shape = SIMD4<UInt32>(
      UInt32(tensor.shape[0]), UInt32(tensor.shape[1]), UInt32(tensor.shape[2]), UInt32(pairCount))
    var offset = UInt32(positionOffset)
    var ropeBase = base
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 2)
    encoder.setBytes(&offset, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&ropeBase, length: MemoryLayout<Float>.stride, index: 4)
    encoder.dispatchThreads(
      MTLSize(width: workItemCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: "RoPE", message: commandBuffer.error?.localizedDescription ?? "unknown error")
    }
    let values = Array(
      UnsafeBufferPointer(
        start: output.contents().bindMemory(to: Float.self, capacity: tensor.elementCount),
        count: tensor.elementCount
      ))
    return try FloatTensor(values, shape: tensor.shape)
  }
}
