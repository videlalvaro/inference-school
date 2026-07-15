import InferenceSchoolCore

public enum P013EmbeddingSolution {
  public static func apply(
    tokenIDs: [Int],
    table: FloatTensor
  ) throws -> EmbeddingLookupResult {
    try EmbeddingLookupContract.validate(tokenIDs: tokenIDs, table: table)
    let vocabularySize = table.shape[0]
    let embeddingDimension = table.shape[1]
    var embeddings: [Float] = []
    embeddings.reserveCapacity(tokenIDs.count * embeddingDimension)
    for token in tokenIDs {
      let rowStart = token * embeddingDimension
      embeddings.append(contentsOf: table.storage[rowStart..<(rowStart + embeddingDimension)])
    }

    var logits = Array(repeating: Float.zero, count: tokenIDs.count * vocabularySize)
    for sequence in tokenIDs.indices {
      for vocabulary in 0..<vocabularySize {
        var sum: Float = 0
        for feature in 0..<embeddingDimension {
          sum +=
            embeddings[sequence * embeddingDimension + feature]
            * table.storage[vocabulary * embeddingDimension + feature]
        }
        logits[sequence * vocabularySize + vocabulary] = sum
      }
    }
    return EmbeddingLookupResult(
      embeddings: try FloatTensor(
        embeddings,
        shape: [tokenIDs.count, embeddingDimension]
      ),
      logits: try FloatTensor(logits, shape: [tokenIDs.count, vocabularySize])
    )
  }
}
