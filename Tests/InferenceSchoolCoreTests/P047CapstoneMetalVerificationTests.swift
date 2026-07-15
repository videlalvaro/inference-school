import InferenceSchoolCore
import InferenceSchoolSolutions
import Metal
import XCTest

final class P047CapstoneMetalVerificationTests: XCTestCase {
  func testCapstoneExecutesMetalVerificationSlice() throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
      throw XCTSkip("Metal is unavailable on this test host.")
    }
    let capstoneRequest = try P047CapstoneFixture.makeRequest(
      maxNewTokens: 2, seed: 47, includeMetalVerification: true)
    let capstoneReport = try P047CapstoneSolution.run(capstoneRequest)
    try P047CapstoneContract.validate(capstoneReport, for: capstoneRequest)
    XCTAssertEqual(capstoneReport.metalVerification.status, .completed)
    XCTAssertTrue(capstoneReport.metalVerification.parityPassed)
    XCTAssertEqual(capstoneReport.metalVerification.captures.count, 5)
    XCTAssertEqual(capstoneReport.metalVerification.resources?.dispatchCount, 3)
    XCTAssertEqual(capstoneReport.metalVerification.resources?.hostWaitCount, 3)
  }
}