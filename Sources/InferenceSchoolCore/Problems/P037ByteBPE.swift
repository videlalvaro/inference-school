import Foundation

public enum ByteBPEError: Error, Equatable, LocalizedError {
  case invalidTokenID(Int)
  case duplicateTokenID(Int)
  case emptyTokenBytes(Int)
  case duplicateTokenBytes([UInt8])
  case invalidSpecialTokens(bos: Int, eos: Int)
  case missingSpecialToken(Int)
  case invalidMergeRank(Int)
  case duplicateMergePair(left: Int, right: Int)
  case duplicateMergeRank(Int)
  case missingMergeToken(id: Int)
  case mergeBytesMismatch(result: Int)
  case invalidUnknownToken(Int)
  case missingByteToken(byte: UInt8, byteIndex: Int)
  case invalidUTF8

  public var errorDescription: String? {
    switch self {
    case .invalidTokenID(let id):
      "Token IDs must be nonnegative; received \(id)."
    case .duplicateTokenID(let id):
      "Token ID \(id) appears more than once."
    case .emptyTokenBytes(let id):
      "Token ID \(id) must own at least one byte."
    case .duplicateTokenBytes(let bytes):
      "Vocabulary byte sequence \(bytes) appears more than once."
    case .invalidSpecialTokens(let bos, let eos):
      "BOS and EOS IDs must be distinct; both were configured as \(bos == eos ? bos : eos)."
    case .missingSpecialToken(let id):
      "Special token ID \(id) is absent from the vocabulary."
    case .invalidMergeRank(let rank):
      "Merge ranks must be nonnegative; received \(rank)."
    case .duplicateMergePair(let left, let right):
      "Merge pair (\(left), \(right)) appears more than once."
    case .duplicateMergeRank(let rank):
      "Merge rank \(rank) appears more than once."
    case .missingMergeToken(let id):
      "Merge rule references missing token ID \(id)."
    case .mergeBytesMismatch(let result):
      "Merge result token \(result) does not equal the left bytes followed by the right bytes."
    case .invalidUnknownToken(let id):
      "Unknown-byte token ID \(id) is absent from the vocabulary."
    case .missingByteToken(let byte, let byteIndex):
      "No one-byte vocabulary token exists for byte \(byte) at UTF-8 byte index \(byteIndex)."
    case .invalidUTF8:
      "Decoded token bytes are not valid UTF-8."
    }
  }
}

public struct BPEVocabularyToken: Sendable, Equatable {
  public let id: Int
  public let bytes: [UInt8]

  public init(id: Int, bytes: [UInt8]) {
    self.id = id
    self.bytes = bytes
  }
}

public struct BPEPair: Sendable, Hashable {
  public let left: Int
  public let right: Int

  public init(left: Int, right: Int) {
    self.left = left
    self.right = right
  }
}

public struct BPEMergeRule: Sendable, Equatable {
  public let left: Int
  public let right: Int
  public let result: Int
  public let rank: Int

  public init(left: Int, right: Int, result: Int, rank: Int) {
    self.left = left
    self.right = right
    self.result = result
    self.rank = rank
  }
}

public enum BPEUnknownBytePolicy: Sendable, Equatable {
  case error
  case token(Int)
}

public struct BPEEncodingOptions: Sendable, Equatable {
  public let addBeginningOfSequence: Bool
  public let addEndOfSequence: Bool

  public init(addBeginningOfSequence: Bool = false, addEndOfSequence: Bool = false) {
    self.addBeginningOfSequence = addBeginningOfSequence
    self.addEndOfSequence = addEndOfSequence
  }
}

public struct ByteBPETokenizer: Sendable, Equatable {
  public let tokensByID: [Int: [UInt8]]
  public let byteTokenIDs: [UInt8: Int]
  public let mergeRules: [BPEPair: BPEMergeRule]
  public let beginningOfSequenceTokenID: Int
  public let endOfSequenceTokenID: Int
  public let unknownBytePolicy: BPEUnknownBytePolicy

  public init(
    vocabulary: [BPEVocabularyToken],
    merges: [BPEMergeRule],
    beginningOfSequenceTokenID: Int,
    endOfSequenceTokenID: Int,
    unknownBytePolicy: BPEUnknownBytePolicy = .error
  ) throws {
    guard beginningOfSequenceTokenID != endOfSequenceTokenID else {
      throw ByteBPEError.invalidSpecialTokens(
        bos: beginningOfSequenceTokenID, eos: endOfSequenceTokenID)
    }
    var byID: [Int: [UInt8]] = [:]
    var byBytes: [[UInt8]: Int] = [:]
    var byteIDs: [UInt8: Int] = [:]
    for token in vocabulary {
      guard token.id >= 0 else { throw ByteBPEError.invalidTokenID(token.id) }
      guard !token.bytes.isEmpty else { throw ByteBPEError.emptyTokenBytes(token.id) }
      guard byID[token.id] == nil else { throw ByteBPEError.duplicateTokenID(token.id) }
      guard byBytes[token.bytes] == nil else {
        throw ByteBPEError.duplicateTokenBytes(token.bytes)
      }
      byID[token.id] = token.bytes
      byBytes[token.bytes] = token.id
      if token.bytes.count == 1 { byteIDs[token.bytes[0]] = token.id }
    }
    for id in [beginningOfSequenceTokenID, endOfSequenceTokenID] where byID[id] == nil {
      throw ByteBPEError.missingSpecialToken(id)
    }
    if case .token(let id) = unknownBytePolicy, byID[id] == nil {
      throw ByteBPEError.invalidUnknownToken(id)
    }
    var rules: [BPEPair: BPEMergeRule] = [:]
    var ranks: Set<Int> = []
    for rule in merges {
      guard rule.rank >= 0 else { throw ByteBPEError.invalidMergeRank(rule.rank) }
      let pair = BPEPair(left: rule.left, right: rule.right)
      guard rules[pair] == nil else {
        throw ByteBPEError.duplicateMergePair(left: rule.left, right: rule.right)
      }
      guard ranks.insert(rule.rank).inserted else {
        throw ByteBPEError.duplicateMergeRank(rule.rank)
      }
      guard let left = byID[rule.left] else {
        throw ByteBPEError.missingMergeToken(id: rule.left)
      }
      guard let right = byID[rule.right] else {
        throw ByteBPEError.missingMergeToken(id: rule.right)
      }
      guard let result = byID[rule.result] else {
        throw ByteBPEError.missingMergeToken(id: rule.result)
      }
      guard result == left + right else {
        throw ByteBPEError.mergeBytesMismatch(result: rule.result)
      }
      rules[pair] = rule
    }
    self.tokensByID = byID
    self.byteTokenIDs = byteIDs
    self.mergeRules = rules
    self.beginningOfSequenceTokenID = beginningOfSequenceTokenID
    self.endOfSequenceTokenID = endOfSequenceTokenID
    self.unknownBytePolicy = unknownBytePolicy
  }

  public func tokenBytes(for id: Int) throws -> [UInt8] {
    guard let bytes = tokensByID[id] else { throw ByteBPEError.invalidTokenID(id) }
    return bytes
  }

  public func initialTokenID(for byte: UInt8, at byteIndex: Int) throws -> Int {
    if let id = byteTokenIDs[byte] { return id }
    switch unknownBytePolicy {
    case .error:
      throw ByteBPEError.missingByteToken(byte: byte, byteIndex: byteIndex)
    case .token(let id):
      return id
    }
  }
}

public typealias BPEEncodeImplementation = (
  _ tokenizer: ByteBPETokenizer,
  _ text: String,
  _ options: BPEEncodingOptions
) throws -> [Int]
public typealias BPEDecodeBytesImplementation = (
  _ tokenizer: ByteBPETokenizer,
  _ tokenIDs: [Int],
  _ skipSpecialTokens: Bool
) throws -> [UInt8]
public typealias BPEDecodeTextImplementation = (
  _ tokenizer: ByteBPETokenizer,
  _ tokenIDs: [Int],
  _ skipSpecialTokens: Bool
) throws -> String

public enum P037ByteBPEFixture {
  public static func makeTokenizer() throws -> ByteBPETokenizer {
    var vocabulary = (0...255).map {
      BPEVocabularyToken(id: $0, bytes: [UInt8($0)])
    }
    vocabulary.append(BPEVocabularyToken(id: 256, bytes: Array("<BOS>".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 257, bytes: Array("<EOS>".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 258, bytes: Array("th".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 259, bytes: Array("the".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 260, bytes: Array(" the".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 261, bytes: Array("é".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 262, bytes: Array("ll".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 263, bytes: Array("ell".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 264, bytes: Array("hell".utf8)))
    vocabulary.append(BPEVocabularyToken(id: 265, bytes: Array("hello".utf8)))
    let merges = [
      BPEMergeRule(left: 116, right: 104, result: 258, rank: 0),
      BPEMergeRule(left: 258, right: 101, result: 259, rank: 1),
      BPEMergeRule(left: 32, right: 259, result: 260, rank: 2),
      BPEMergeRule(left: 195, right: 169, result: 261, rank: 3),
      BPEMergeRule(left: 108, right: 108, result: 262, rank: 4),
      BPEMergeRule(left: 101, right: 262, result: 263, rank: 5),
      BPEMergeRule(left: 104, right: 263, result: 264, rank: 6),
      BPEMergeRule(left: 264, right: 111, result: 265, rank: 7),
    ]
    return try ByteBPETokenizer(
      vocabulary: vocabulary,
      merges: merges,
      beginningOfSequenceTokenID: 256,
      endOfSequenceTokenID: 257,
      unknownBytePolicy: .error)
  }
}

public enum P037ByteBPEJudge {
  public static func evaluate(
    encode: BPEEncodeImplementation,
    decodeBytes: BPEDecodeBytesImplementation,
    decodeText: BPEDecodeTextImplementation
  ) -> JudgeReport {
    var passed = 0
    var failures: [JudgeFailure] = []
    do {
      let tokenizer = try P037ByteBPEFixture.makeTokenizer()
      passed += expectValue(
        name: "ranked merges and leftmost ties",
        expected: [259, 260, 32, 258, 258],
        failures: &failures
      ) {
        try encode(tokenizer, "the the thth", BPEEncodingOptions())
      }
      passed += expectValue(
        name: "Unicode starts from UTF-8 bytes",
        expected: [99, 97, 102, 261],
        failures: &failures
      ) {
        try encode(tokenizer, "café", BPEEncodingOptions())
      }
      passed += expectValue(
        name: "empty input with BOS and EOS",
        expected: [256, 257],
        failures: &failures
      ) {
        try encode(tokenizer, "", BPEEncodingOptions(
          addBeginningOfSequence: true, addEndOfSequence: true))
      }
      let text = "hello 👋"
      let encoded = try encode(tokenizer, text, BPEEncodingOptions(
        addBeginningOfSequence: true, addEndOfSequence: true))
      if try decodeText(tokenizer, encoded, true) == text {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "text round trip", message: "decoded text does not match UTF-8 input"))
      }
      if try decodeBytes(tokenizer, [255], false) == [255] {
        passed += 1
      } else {
        failures.append(JudgeFailure(
          caseName: "raw invalid UTF-8 byte", message: "raw-byte decode must preserve 0xff"))
      }
      passed += expectError(name: "text decode rejects invalid UTF-8", failures: &failures) {
        _ = try decodeText(tokenizer, [255], false)
      }
      passed += expectError(name: "reject invalid token ID", failures: &failures) {
        _ = try decodeBytes(tokenizer, [999], false)
      }
      let sparse = try ByteBPETokenizer(
        vocabulary: [
          BPEVocabularyToken(id: 0, bytes: [0]),
          BPEVocabularyToken(id: 1, bytes: Array("<B>".utf8)),
          BPEVocabularyToken(id: 2, bytes: Array("<E>".utf8)),
        ],
        merges: [],
        beginningOfSequenceTokenID: 1,
        endOfSequenceTokenID: 2,
        unknownBytePolicy: .error)
      passed += expectError(name: "unknown byte policy is explicit", failures: &failures) {
        _ = try encode(sparse, "A", BPEEncodingOptions())
      }
    } catch {
      failures.append(JudgeFailure(caseName: "judge execution", message: error.localizedDescription))
    }
    return JudgeReport(passedCaseCount: passed, totalCaseCount: 8, failures: failures)
  }

  private static func expectValue<T: Equatable>(
    name: String,
    expected: T,
    failures: inout [JudgeFailure],
    operation: () throws -> T
  ) -> Int {
    do {
      let actual = try operation()
      guard actual == expected else {
        failures.append(JudgeFailure(
          caseName: name, message: "expected \(expected), received \(actual)"))
        return 0
      }
      return 1
    } catch {
      failures.append(JudgeFailure(caseName: name, message: error.localizedDescription))
      return 0
    }
  }

  private static func expectError(
    name: String,
    failures: inout [JudgeFailure],
    operation: () throws -> Void
  ) -> Int {
    do {
      try operation()
      failures.append(JudgeFailure(
        caseName: name, message: "expected an error, but the tokenizer returned"))
      return 0
    } catch {
      return 1
    }
  }
}