import Metal

public final class MetalEmbeddingPipeline {
  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let lookupPipeline: any MTLComputePipelineState
  private let unembeddingPipeline: any MTLComputePipelineState

  public init(
    source: String,
    lookupFunctionName: String = "embedding_lookup",
    unembeddingFunctionName: String = "tied_unembedding"
  ) throws {
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
        operation: "embedding lookup",
        message: error.localizedDescription
      )
    }
    guard let lookupFunction = library.makeFunction(name: lookupFunctionName) else {
      throw MetalNeuralOperatorError.functionNotFound(lookupFunctionName)
    }
    guard let unembeddingFunction = library.makeFunction(name: unembeddingFunctionName) else {
      throw MetalNeuralOperatorError.functionNotFound(unembeddingFunctionName)
    }
    do {
      lookupPipeline = try device.makeComputePipelineState(function: lookupFunction)
      unembeddingPipeline = try device.makeComputePipelineState(function: unembeddingFunction)
    } catch {
      throw MetalNeuralOperatorError.pipelineCreationFailed(
        operation: "embedding lookup",
        message: error.localizedDescription
      )
    }
    self.device = device
    self.commandQueue = commandQueue
  }

  public func apply(_ tokenIDs: [Int], table: FloatTensor) throws -> EmbeddingLookupResult {
    try EmbeddingLookupContract.validate(tokenIDs: tokenIDs, table: table)
    let vocabularySize = table.shape[0]
    let embeddingDimension = table.shape[1]
    guard vocabularySize <= UInt32.max,
      embeddingDimension <= UInt32.max,
      tokenIDs.count <= UInt32.max
    else {
      throw MetalNeuralOperatorError.dimensionsTooLarge
    }
    guard !tokenIDs.isEmpty else {
      return EmbeddingLookupResult(
        embeddings: try FloatTensor([], shape: [0, embeddingDimension]),
        logits: try FloatTensor([], shape: [0, vocabularySize])
      )
    }

    let metalTokenIDs = tokenIDs.map(UInt32.init)
    let tableByteCount = table.elementCount * MemoryLayout<Float>.stride
    let tokenByteCount = metalTokenIDs.count * MemoryLayout<UInt32>.stride
    let embeddingCount = tokenIDs.count * embeddingDimension
    let logitsCount = tokenIDs.count * vocabularySize
    guard
      let tableBuffer = device.makeBuffer(
        bytes: table.storage,
        length: tableByteCount,
        options: .storageModeShared
      )
    else {
      throw MetalNeuralOperatorError.bufferCreationFailed("embedding table")
    }
    guard
      let tokenBuffer = device.makeBuffer(
        bytes: metalTokenIDs,
        length: tokenByteCount,
        options: .storageModeShared
      )
    else {
      throw MetalNeuralOperatorError.bufferCreationFailed("token IDs")
    }
    guard
      let embeddingBuffer = device.makeBuffer(
        length: embeddingCount * MemoryLayout<Float>.stride,
        options: .storageModeShared
      )
    else {
      throw MetalNeuralOperatorError.bufferCreationFailed("gathered embeddings")
    }
    guard
      let logitsBuffer = device.makeBuffer(
        length: logitsCount * MemoryLayout<Float>.stride,
        options: .storageModeShared
      )
    else {
      throw MetalNeuralOperatorError.bufferCreationFailed("unembedding logits")
    }
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      throw MetalNeuralOperatorError.commandCreationFailed
    }

    var shape = SIMD4<UInt32>(
      UInt32(vocabularySize),
      UInt32(embeddingDimension),
      UInt32(tokenIDs.count),
      0
    )
    guard let lookupEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw MetalNeuralOperatorError.commandCreationFailed
    }
    lookupEncoder.setComputePipelineState(lookupPipeline)
    lookupEncoder.setBuffer(tableBuffer, offset: 0, index: 0)
    lookupEncoder.setBuffer(tokenBuffer, offset: 0, index: 1)
    lookupEncoder.setBuffer(embeddingBuffer, offset: 0, index: 2)
    lookupEncoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
    lookupEncoder.dispatchThreads(
      MTLSize(width: embeddingCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(256, lookupPipeline.maxTotalThreadsPerThreadgroup),
        height: 1,
        depth: 1
      )
    )
    lookupEncoder.endEncoding()

    guard let unembeddingEncoder = commandBuffer.makeComputeCommandEncoder() else {
      throw MetalNeuralOperatorError.commandCreationFailed
    }
    unembeddingEncoder.setComputePipelineState(unembeddingPipeline)
    unembeddingEncoder.setBuffer(embeddingBuffer, offset: 0, index: 0)
    unembeddingEncoder.setBuffer(tableBuffer, offset: 0, index: 1)
    unembeddingEncoder.setBuffer(logitsBuffer, offset: 0, index: 2)
    unembeddingEncoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
    unembeddingEncoder.dispatchThreads(
      MTLSize(width: logitsCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(256, unembeddingPipeline.maxTotalThreadsPerThreadgroup),
        height: 1,
        depth: 1
      )
    )
    unembeddingEncoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalNeuralOperatorError.commandFailed(
        operation: "embedding lookup",
        message: commandBuffer.error?.localizedDescription ?? "unknown error"
      )
    }

    let embeddings = Array(
      UnsafeBufferPointer(
        start: embeddingBuffer.contents().bindMemory(to: Float.self, capacity: embeddingCount),
        count: embeddingCount
      ))
    let logits = Array(
      UnsafeBufferPointer(
        start: logitsBuffer.contents().bindMemory(to: Float.self, capacity: logitsCount),
        count: logitsCount
      ))
    return EmbeddingLookupResult(
      embeddings: try FloatTensor(embeddings, shape: [tokenIDs.count, embeddingDimension]),
      logits: try FloatTensor(logits, shape: [tokenIDs.count, vocabularySize])
    )
  }
}
