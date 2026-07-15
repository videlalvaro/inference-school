import InferenceSchoolCore

public enum P013EmbeddingExercise {
  public static func apply(
    tokenIDs: [Int],
    table: FloatTensor
  ) throws -> EmbeddingLookupResult {
    try EmbeddingLookupContract.validate(tokenIDs: tokenIDs, table: table)
    let embeddings = try FloatTensor(
      Array(repeating: 0, count: tokenIDs.count * table.shape[1]),
      shape: [tokenIDs.count, table.shape[1]]
    )
    let logits = try FloatTensor(
      Array(repeating: 0, count: tokenIDs.count * table.shape[0]),
      shape: [tokenIDs.count, table.shape[0]]
    )
    return EmbeddingLookupResult(embeddings: embeddings, logits: logits)
  }
}
