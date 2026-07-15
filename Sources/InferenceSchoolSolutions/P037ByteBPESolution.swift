import Foundation
import InferenceSchoolCore

public enum P037ByteBPESolution {
  public static func encode(
    tokenizer: ByteBPETokenizer,
    text: String,
    options: BPEEncodingOptions
  ) throws -> [Int] {
    var tokenIDs = try Array(text.utf8).enumerated().map {
      try tokenizer.initialTokenID(for: $0.element, at: $0.offset)
    }
    while tokenIDs.count >= 2 {
      var bestIndex: Int?
      var bestRule: BPEMergeRule?
      for index in 0..<(tokenIDs.count - 1) {
        let pair = BPEPair(left: tokenIDs[index], right: tokenIDs[index + 1])
        guard let rule = tokenizer.mergeRules[pair] else { continue }
        if bestRule == nil || rule.rank < bestRule!.rank {
          bestIndex = index
          bestRule = rule
        }
      }
      guard let index = bestIndex, let rule = bestRule else { break }
      tokenIDs.replaceSubrange(index...(index + 1), with: [rule.result])
    }
    if options.addBeginningOfSequence {
      tokenIDs.insert(tokenizer.beginningOfSequenceTokenID, at: 0)
    }
    if options.addEndOfSequence {
      tokenIDs.append(tokenizer.endOfSequenceTokenID)
    }
    return tokenIDs
  }

  public static func decodeBytes(
    tokenizer: ByteBPETokenizer,
    tokenIDs: [Int],
    skipSpecialTokens: Bool
  ) throws -> [UInt8] {
    var bytes: [UInt8] = []
    for tokenID in tokenIDs {
      if skipSpecialTokens,
        tokenID == tokenizer.beginningOfSequenceTokenID
          || tokenID == tokenizer.endOfSequenceTokenID
      {
        continue
      }
      bytes.append(contentsOf: try tokenizer.tokenBytes(for: tokenID))
    }
    return bytes
  }

  public static func decodeText(
    tokenizer: ByteBPETokenizer,
    tokenIDs: [Int],
    skipSpecialTokens: Bool
  ) throws -> String {
    let bytes = try decodeBytes(
      tokenizer: tokenizer,
      tokenIDs: tokenIDs,
      skipSpecialTokens: skipSpecialTokens)
    guard let text = String(bytes: bytes, encoding: .utf8) else {
      throw ByteBPEError.invalidUTF8
    }
    return text
  }
}