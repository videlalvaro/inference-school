import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P037ByteBPETests: XCTestCase {
  func testCanonicalTokenizerPassesJudge() {
    let report = P037ByteBPEJudge.evaluate(
      encode: P037ByteBPESolution.encode,
      decodeBytes: P037ByteBPESolution.decodeBytes,
      decodeText: P037ByteBPESolution.decodeText)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testJudgeRejectsByteOnlyTokenizerWithoutMerges() {
    let report = P037ByteBPEJudge.evaluate(
      encode: { tokenizer, text, options in
        var ids = try Array(text.utf8).enumerated().map {
          try tokenizer.initialTokenID(for: $0.element, at: $0.offset)
        }
        if options.addBeginningOfSequence {
          ids.insert(tokenizer.beginningOfSequenceTokenID, at: 0)
        }
        if options.addEndOfSequence { ids.append(tokenizer.endOfSequenceTokenID) }
        return ids
      },
      decodeBytes: P037ByteBPESolution.decodeBytes,
      decodeText: P037ByteBPESolution.decodeText)
    XCTAssertFalse(report.isPassing)
  }

  func testUnicodeRoundTripAndInvalidUTF8Policies() throws {
    let tokenizer = try P037ByteBPEFixture.makeTokenizer()
    let input = "café 👋"
    let ids = try P037ByteBPESolution.encode(
      tokenizer: tokenizer,
      text: input,
      options: BPEEncodingOptions(
        addBeginningOfSequence: true, addEndOfSequence: true))
    XCTAssertEqual(try P037ByteBPESolution.decodeText(
      tokenizer: tokenizer, tokenIDs: ids, skipSpecialTokens: true), input)
    XCTAssertEqual(try P037ByteBPESolution.decodeBytes(
      tokenizer: tokenizer, tokenIDs: [255], skipSpecialTokens: false), [255])
    XCTAssertThrowsError(try P037ByteBPESolution.decodeText(
      tokenizer: tokenizer, tokenIDs: [255], skipSpecialTokens: false))
    XCTAssertThrowsError(try P037ByteBPESolution.decodeBytes(
      tokenizer: tokenizer, tokenIDs: [10_000], skipSpecialTokens: false))
  }

  func testEmptyInputSpecialTokenPolicyIsExplicit() throws {
    let tokenizer = try P037ByteBPEFixture.makeTokenizer()
    XCTAssertEqual(try P037ByteBPESolution.encode(
      tokenizer: tokenizer, text: "", options: BPEEncodingOptions()), [])
    XCTAssertEqual(try P037ByteBPESolution.encode(
      tokenizer: tokenizer,
      text: "",
      options: BPEEncodingOptions(
        addBeginningOfSequence: true, addEndOfSequence: true)), [256, 257])
  }
}