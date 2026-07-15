import Metal

public final class MetalFusedQ4GEMVPipeline {
  public static let threadgroupWidth = 256

  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState

  public init(source: String, functionName: String = "fused_q4_gemv") throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do { library = try device.makeLibrary(source: source, options: nil) } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: "fused Q4 GEMV", message: error.localizedDescription)
    }
    guard let function = library.makeFunction(name: functionName) else {
      throw MetalNeuralOperatorError.functionNotFound(functionName)
    }
    do { pipeline = try device.makeComputePipelineState(function: function) } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "fused Q4 GEMV", message: error.localizedDescription)
    }
    guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else {
      throw MetalNeuralOperatorError.unsupportedThreadgroupWidth(
        required: Self.threadgroupWidth,
        maximum: pipeline.maxTotalThreadsPerThreadgroup)
    }
    self.device = device
    self.commandQueue = commandQueue
  }

  public func multiply(
    _ weights: GroupwiseQ4WeightMatrix,
    _ input: FloatTensor
  ) throws -> FusedQ4GEMVResult {
    guard input.rank == 1 else {
      throw TensorError.rankMismatch(expected: 1, actual: input.rank)
    }
    guard input.shape[0] == weights.inputChannels else {
      throw DenseLinearAlgebraError.innerDimensionMismatch(
        operation: "Metal fused Q4 GEMV", lhs: weights.inputChannels, rhs: input.shape[0])
    }
    for (index, value) in input.storage.enumerated() where !value.isFinite {
      throw WeightQuantizationError.nonFiniteValue(index: index, value: value)
    }
    guard weights.outputChannels <= UInt32.max,
      weights.inputChannels <= UInt32.max,
      weights.groupSize <= UInt32.max,
      weights.groupsPerOutputChannel <= UInt32.max
    else { throw MetalNeuralOperatorError.dimensionsTooLarge }
    guard weights.outputChannels > 0 else {
      return FusedQ4GEMVResult(
        output: try FloatTensor([], shape: [0]),
        logicalWeightBytes: weights.allocatedBytes,
        temporaryWeightBytes: 0)
    }
    guard weights.inputChannels > 0 else {
      return FusedQ4GEMVResult(
        output: try FloatTensor(
          Array(repeating: 0, count: weights.outputChannels), shape: [weights.outputChannels]),
        logicalWeightBytes: weights.allocatedBytes,
        temporaryWeightBytes: 0)
    }

    guard let packedBuffer = weights.packedValues.withUnsafeBytes({ bytes in
      device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
    }) else { throw MetalNeuralOperatorError.bufferCreationFailed("packed Q4 weights") }
    guard let scaleBuffer = device.makeBuffer(
      bytes: weights.scales,
      length: weights.scales.count * MemoryLayout<Float>.stride,
      options: .storageModeShared)
    else { throw MetalNeuralOperatorError.bufferCreationFailed("Q4 scales") }
    guard let inputBuffer = device.makeBuffer(
      bytes: input.storage,
      length: input.elementCount * MemoryLayout<Float>.stride,
      options: .storageModeShared)
    else { throw MetalNeuralOperatorError.bufferCreationFailed("GEMV input") }
    guard let outputBuffer = device.makeBuffer(
      length: weights.outputChannels * MemoryLayout<Float>.stride,
      options: .storageModeShared)
    else { throw MetalNeuralOperatorError.bufferCreationFailed("GEMV output") }
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else { throw MetalNeuralOperatorError.commandCreationFailed }

    var shape = SIMD4<UInt32>(
      UInt32(weights.outputChannels),
      UInt32(weights.inputChannels),
      UInt32(weights.groupSize),
      weights.format.rawValue)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(packedBuffer, offset: 0, index: 0)
    encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
    encoder.setBuffer(inputBuffer, offset: 0, index: 2)
    encoder.setBuffer(outputBuffer, offset: 0, index: 3)
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 4)
    encoder.dispatchThreadgroups(
      MTLSize(width: weights.outputChannels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: "fused Q4 GEMV",
        message: commandBuffer.error?.localizedDescription ?? "unknown error")
    }

    let values = Array(UnsafeBufferPointer(
      start: outputBuffer.contents().bindMemory(
        to: Float.self, capacity: weights.outputChannels),
      count: weights.outputChannels))
    return FusedQ4GEMVResult(
      output: try FloatTensor(values, shape: [weights.outputChannels]),
      logicalWeightBytes: weights.allocatedBytes,
      temporaryWeightBytes: 0)
  }
}