import Metal

public struct MetalFusedQKVExecution: Sendable, Equatable {
  public let result: FusedQKVResult
  public let allocatedBufferBytes: Int
  public let hostToDeviceBytes: Int
  public let deviceToHostBytes: Int
  public let dispatchCount: Int
  public let commandBufferCount: Int
  public let hostWaitCount: Int

  public init(
    result: FusedQKVResult,
    allocatedBufferBytes: Int,
    hostToDeviceBytes: Int,
    deviceToHostBytes: Int,
    dispatchCount: Int,
    commandBufferCount: Int,
    hostWaitCount: Int
  ) {
    self.result = result
    self.allocatedBufferBytes = allocatedBufferBytes
    self.hostToDeviceBytes = hostToDeviceBytes
    self.deviceToHostBytes = deviceToHostBytes
    self.dispatchCount = dispatchCount
    self.commandBufferCount = commandBufferCount
    self.hostWaitCount = hostWaitCount
  }
}

public final class MetalFusedQKVPipeline {
  public static let threadgroupWidth = 256

  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState

  public init(source: String, functionName: String = "fused_rmsnorm_qkv") throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalNeuralOperatorError.noDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalNeuralOperatorError.commandQueueCreationFailed
    }
    let library: any MTLLibrary
    do {
      library = try device.makeLibrary(source: source, options: nil)
    } catch {
      throw MetalNeuralOperatorError.libraryCreationFailed(
        operation: "fused RMSNorm plus Q/K/V", message: error.localizedDescription)
    }
    guard let function = library.makeFunction(name: functionName) else {
      throw MetalNeuralOperatorError.functionNotFound(functionName)
    }
    do {
      pipeline = try device.makeComputePipelineState(function: function)
    } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "fused RMSNorm plus Q/K/V", message: error.localizedDescription)
    }
    guard pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth else {
      throw MetalNeuralOperatorError.unsupportedThreadgroupWidth(
        required: Self.threadgroupWidth,
        maximum: pipeline.maxTotalThreadsPerThreadgroup)
    }
    self.device = device
    self.commandQueue = commandQueue
  }

  public func project(_ request: FusedQKVRequest) throws -> FusedQKVResult {
    try run(request).result
  }

  public func run(_ request: FusedQKVRequest) throws -> MetalFusedQKVExecution {
    try P043FusedQKVContract.validate(request)
    let sequence = request.input.shape[0]
    let model = request.configuration.modelDimension
    let query = request.configuration.queryProjectionDimension
    let keyValue = request.configuration.keyValueProjectionDimension
    let floatBytes = MemoryLayout<Float>.stride

    func inputBuffer(_ values: [Float], name: String) throws -> any MTLBuffer {
      guard let buffer = device.makeBuffer(
        bytes: values,
        length: values.count * floatBytes,
        options: .storageModeShared)
      else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return buffer
    }
    func outputBuffer(count: Int, name: String) throws -> any MTLBuffer {
      guard let buffer = device.makeBuffer(
        length: count * floatBytes,
        options: .storageModeShared)
      else { throw MetalNeuralOperatorError.bufferCreationFailed(name) }
      return buffer
    }

    let input = try inputBuffer(request.input.storage, name: "fused QKV input")
    let gamma = try inputBuffer(request.gamma.storage, name: "fused QKV gamma")
    let queryWeights = try inputBuffer(
      request.queryWeights.storage, name: "fused query weights")
    let keyWeights = try inputBuffer(request.keyWeights.storage, name: "fused key weights")
    let valueWeights = try inputBuffer(
      request.valueWeights.storage, name: "fused value weights")
    let queryOutput = try outputBuffer(count: sequence * query, name: "fused query output")
    let keyOutput = try outputBuffer(count: sequence * keyValue, name: "fused key output")
    let valueOutput = try outputBuffer(count: sequence * keyValue, name: "fused value output")

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else { throw MetalNeuralOperatorError.commandCreationFailed }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(gamma, offset: 0, index: 1)
    encoder.setBuffer(queryWeights, offset: 0, index: 2)
    encoder.setBuffer(keyWeights, offset: 0, index: 3)
    encoder.setBuffer(valueWeights, offset: 0, index: 4)
    encoder.setBuffer(queryOutput, offset: 0, index: 5)
    encoder.setBuffer(keyOutput, offset: 0, index: 6)
    encoder.setBuffer(valueOutput, offset: 0, index: 7)
    var shape = SIMD4<UInt32>(UInt32(sequence), UInt32(model), UInt32(query), UInt32(keyValue))
    var epsilon = request.epsilon
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 8)
    encoder.setBytes(&epsilon, length: MemoryLayout<Float>.stride, index: 9)
    encoder.dispatchThreadgroups(
      MTLSize(width: sequence, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: "fused RMSNorm plus Q/K/V",
        message: commandBuffer.error?.localizedDescription ?? "unknown error")
    }

    func values(_ buffer: any MTLBuffer, count: Int) -> [Float] {
      Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: Float.self, capacity: count),
        count: count))
    }
    let result = FusedQKVResult(
      queries: try FloatTensor(
        values(queryOutput, count: sequence * query),
        shape: [sequence, request.configuration.queryHeadCount, request.configuration.headDimension]),
      keys: try FloatTensor(
        values(keyOutput, count: sequence * keyValue),
        shape: [sequence, request.configuration.keyValueHeadCount, request.configuration.headDimension]),
      values: try FloatTensor(
        values(valueOutput, count: sequence * keyValue),
        shape: [sequence, request.configuration.keyValueHeadCount, request.configuration.headDimension]))
    try P043FusedQKVContract.validate(result, for: request)

    let hostToDeviceBytes =
      (request.input.elementCount + request.gamma.elementCount + request.queryWeights.elementCount
        + request.keyWeights.elementCount + request.valueWeights.elementCount) * floatBytes
    let deviceToHostBytes = sequence * (query + 2 * keyValue) * floatBytes
    return MetalFusedQKVExecution(
      result: result,
      allocatedBufferBytes: hostToDeviceBytes + deviceToHostBytes,
      hostToDeviceBytes: hostToDeviceBytes,
      deviceToHostBytes: deviceToHostBytes,
      dispatchCount: 1,
      commandBufferCount: 1,
      hostWaitCount: 1)
  }
}