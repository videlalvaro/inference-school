import Darwin
import Foundation
import InferenceSchoolCore
import InferenceSchoolExercises
import InferenceSchoolSolutions
import InferenceSchoolRunnerProtocol

private enum CLIError: Error, LocalizedError {
	case invalidArguments(String)
	case unknownCommand(String)
	case unknownProblem(String)

	var errorDescription: String? {
		switch self {
		case let .invalidArguments(message):
			message
		case let .unknownCommand(command):
			"Unknown command '\(command)'."
		case let .unknownProblem(problem):
			"Unknown problem '\(problem)'."
		}
	}
}

private enum CheckStage {
	case all
	case cpu
	case metal
}

private enum CheckOutputFormat: String {
	case text
	case jsonl
}

public struct ProblemRunner: Sendable {
	public let exerciseFiles: [String]
	public let metalFile: String?
	public let cpuCheck: @Sendable (_ useSolution: Bool) -> JudgeReport
	public let metalCheck: (@Sendable (_ useSolution: Bool) throws -> JudgeReport)?

	public init(
		exerciseFiles: [String],
		metalFile: String?,
		cpuCheck: @escaping @Sendable (_ useSolution: Bool) -> JudgeReport,
		metalCheck: (@Sendable (_ useSolution: Bool) throws -> JudgeReport)?
	) {
		self.exerciseFiles = exerciseFiles
		self.metalFile = metalFile
		self.cpuCheck = cpuCheck
		self.metalCheck = metalCheck
	}
}

public enum InferenceSchoolCommand {
	public static func run(arguments: [String]) throws -> Int32 {
		guard let command = arguments.first else {
			printUsage()
			return 0
		}

		switch command {
		case "help", "--help", "-h":
			printUsage()
			return 0
		case "list":
			listProblems()
			return 0
		case "learn":
			let problemID = try requiredProblemID(in: arguments)
			printLearningPath(problemID)
			return 0
		case "show":
			let problemID = try requiredProblemID(in: arguments)
			showProblem(problemID)
			return 0
		case "check":
			return try checkProblem(arguments: arguments)
		case "benchmark":
			try benchmarkProblem(arguments: arguments)
			return 0
		case "profile":
			try profileProblem(arguments: arguments)
			return 0
		case "capstone":
			try runCapstone(arguments: arguments)
			return 0
		default:
			throw CLIError.unknownCommand(command)
		}
	}

	private static func listProblems() {
		print("Available problems")
		for problem in Course.availableProblems {
			print("  \(problem.id)  \(problem.title)")
			print("       engine use: \(problem.engineUse)")
		}
		print("\nFull roadmap: docs/CURRICULUM.md")
	}

	private static func showProblem(_ problemID: String) {
		guard let problem = Course.problem(id: problemID),
			let runner = RuntimeRegistry.runner(for: problemID)
		else {
			return
		}
		print("\(problem.id) - \(problem.title)")
		print("Concepts: \(problem.concept)")
		print("Engine use: \(problem.engineUse)")
		print("Tutorial: \(problem.chapterPath)")
		for file in runner.exerciseFiles {
			print("Exercise: \(file)")
		}
		if let metalFile = runner.metalFile {
			print("Metal: \(metalFile)")
		}
		print("\nRun 'swift run inference-school learn \(problemID)' for the learning sequence.")
	}

	private static func printLearningPath(_ problemID: String) {
		guard let problem = Course.problem(id: problemID),
			let runner = RuntimeRegistry.runner(for: problemID)
		else {
			return
		}
		print("Learn problem \(problem.id): \(problem.title)\n")
		print("1. Read the lesson:\n   \(problem.chapterPath)\n")
		print("2. Implement the Swift exercise:")
		for file in runner.exerciseFiles {
			print("   \(file)")
		}
		print("   swift run inference-school check \(problemID) --cpu\n")
		var nextStep = 3
		if let metalFile = runner.metalFile {
			print("\(nextStep). Implement the Metal exercise:\n   \(metalFile)")
			print("   swift run inference-school check \(problemID) --metal\n")
			nextStep += 1
		}
		print("\(nextStep). Check the complete learner implementation:")
		print("   swift run inference-school check \(problemID)\n")
		nextStep += 1
		print("\(nextStep). Verify the harness with the canonical answer:")
		print("   swift run inference-school check \(problemID) --solution\n")
		if problemID == "047" {
			nextStep += 1
			print("\(nextStep). Run the complete canonical engine report:")
			print("   swift run -c release inference-school capstone --prompt \"ab c.\" --max-tokens 4\n")
		} else if problemID == "044" {
			nextStep += 1
			print("\(nextStep). Profile prefill and decode independently:")
			print("   swift run -c release inference-school profile 044 --prompt-tokens 16 --trials 7 --decode-steps 4\n")
		} else if problemID == "041" {
			nextStep += 1
			print("\(nextStep). Print the chapter's deterministic arena report:")
			print("   swift run inference-school benchmark 041 --tokens 128 --cached-tokens 128\n")
		} else if problemID == "001" || problemID == "006" || problemID == "024"
			|| problemID == "033" || problemID == "043"
		{
			nextStep += 1
			print("\(nextStep). Run the chapter's release benchmark:")
			print("   swift run -c release inference-school benchmark \(problemID)\n")
		}
		print("Orientation and the full workflow:\n   docs/START-HERE.md")
	}

	private static func checkProblem(arguments: [String]) throws -> Int32 {
		let problemIDs = try requiredProblemIDs(in: arguments)
		try validateOptions(
			arguments: arguments,
			commandPrefix: ["check", arguments[1]],
			options: ["--format", "--run-id", "--activity-id"],
			flags: ["--cpu", "--metal", "--solution"]
		)
		let flags = Set(arguments.dropFirst(2))
		let formatValue = try stringOption("--format", in: arguments, default: "text")
		guard let outputFormat = CheckOutputFormat(rawValue: formatValue) else {
			throw CLIError.invalidArguments("--format must be 'text' or 'jsonl'.")
		}
		let requestedRunID = try stringOption("--run-id", in: arguments, default: "")
		let requestedActivityID = try stringOption("--activity-id", in: arguments, default: "")
		guard !(flags.contains("--cpu") && flags.contains("--metal")) else {
			throw CLIError.invalidArguments("Choose either --cpu or --metal, not both.")
		}
		let stage: CheckStage = if flags.contains("--cpu") {
			.cpu
		} else if flags.contains("--metal") {
			.metal
		} else {
			.all
		}
		let useSolution = flags.contains("--solution")
		var isPassing = true
		for problemID in problemIDs {
			guard let activity = RuntimeRegistry.activity(forLessonID: problemID) else {
				throw CLIError.unknownProblem(problemID)
			}
			if flags.contains("--metal"), !activity.supportedStages.contains(.metal) {
				throw CLIError.invalidArguments("Problem \(problemID) has no Metal stage.")
			}
			if problemIDs.count > 1, outputFormat == .text {
				print("Problem \(problemID)")
			}
			let stageIDs: [RunStageID] = switch stage {
			case .cpu:
				[.cpu]
			case .metal:
				[.metal]
			case .all:
				activity.supportedStages
			}
			let request = RunRequest(
				runID: requestedRunID.isEmpty ? UUID().uuidString : requestedRunID,
				lessonID: problemID,
				activityID: requestedActivityID.isEmpty ? activity.id : requestedActivityID,
				mode: currentRunMode,
				stages: stageIDs,
				implementation: useSolution ? .canonical : .learner
			)
			let events = RuntimeCheckExecutor.events(for: request)
			for event in events {
				switch outputFormat {
				case .text:
					renderText(event)
				case .jsonl:
					print(try RunnerJSONL.encode(event))
				}
				if case let .completed(completion) = event.event,
					completion.status != .passed
				{
					isPassing = false
				}
			}
		}

		return isPassing ? 0 : 1
	}

	private static func renderText(_ event: RunEvent) {
		switch event.event {
		case let .judgeReport(report):
			printReport(report)
		case let .diagnostic(diagnostic):
			let label = diagnostic.stageID.map(stageLabel) ?? "Runner"
			print("[FAIL] \(label) setup: \(diagnostic.message)")
		case .accepted, .buildStarted, .buildFinished, .stageStarted,
			.stdout, .stderr, .completed:
			break
		}
	}

	private static func printReport(_ report: RunJudgeReport) {
		let status = report.isPassing ? "PASS" : "FAIL"
		print("[\(status)] \(stageLabel(report.stageID)): \(report.passedCaseCount)/\(report.totalCaseCount) cases")
		for failure in report.failures {
			print("  - \(failure.name): \(failure.message)")
		}
	}

	private static func stageLabel(_ stageID: RunStageID) -> String {
		switch stageID {
		case .cpu: "CPU"
		case .metal: "Metal"
		}
	}

	private static var currentRunMode: RunMode {
#if DEBUG
		.debug
#else
		.release
#endif
	}

	private static func benchmarkProblem(arguments: [String]) throws {
		let problemID = try requiredProblemID(in: arguments)
		switch problemID {
		case "001":
			try benchmarkVectorDot(arguments: arguments)
		case "006":
			try benchmarkRoofline(arguments: arguments)
		case "024":
			try benchmarkKVLayouts(arguments: arguments)
		case "033":
			try benchmarkFusedQ4GEMV(arguments: arguments)
		case "041":
			try reportBufferPlans(arguments: arguments)
		case "043":
			try benchmarkFusedQKV(arguments: arguments)
		case "044":
			try profilePrefillDecode(arguments: arguments, command: "benchmark")
		default:
			throw CLIError.unknownProblem(problemID)
		}
	}

	private static func benchmarkVectorDot(arguments: [String]) throws {
		let size = try integerOption("--size", in: arguments, default: 1_048_576)
		let iterations = try integerOption("--iterations", in: arguments, default: 20)
		guard size > 0, iterations > 0 else {
			throw CLIError.invalidArguments("--size and --iterations must be positive integers.")
		}

		let options = ["--size", "--iterations"]
		let knownArguments = Set(["benchmark", "001"] + options)
		let optionValues = valuesFollowingOptions(options, in: arguments)
		if let unknown = arguments.first(where: {
			!knownArguments.contains($0) && !optionValues.contains($0)
		}) {
			throw CLIError.invalidArguments("Unknown benchmark option '\(unknown)'.")
		}

		let lhs = (0..<size).map { Float(($0 % 31) - 15) / 16 }
		let rhs = (0..<size).map { Float(($0 % 23) - 11) / 12 }
		let metalPipeline = try P001VectorDotSolution.makeMetalPipeline()

#if DEBUG
		print("Note: use 'swift run -c release inference-school benchmark 001' for meaningful timings.")
#endif
		print("Canonical solution, N=\(size), median of \(iterations) iterations")

		let cpu = try measure(iterations: iterations) {
			try P001VectorDotSolution.dot(lhs, rhs)
		}
		printMeasurement(label: "CPU", measurement: cpu, elementCount: size)

		let metal = try measure(iterations: iterations) {
			try metalPipeline.dot(lhs, rhs)
		}
		printMeasurement(label: "Metal end-to-end", measurement: metal, elementCount: size)
		print("Metal timing includes shared-buffer allocation, copies, submission, synchronization, and final CPU reduction.")
	}

	private static func benchmarkRoofline(arguments: [String]) throws {
		let m = try integerOption("--m", in: arguments, default: 64)
		let k = try integerOption("--k", in: arguments, default: 64)
		let n = try integerOption("--n", in: arguments, default: 64)
		let iterations = try integerOption("--iterations", in: arguments, default: 5)
		let peakGFLOPS = try doubleOption("--peak-gflops", in: arguments, default: 1_000)
		let bandwidthGBps = try doubleOption("--bandwidth-gbps", in: arguments, default: 100)
		guard m > 0, k > 0, n > 0, iterations > 0 else {
			throw CLIError.invalidArguments("--m, --k, --n, and --iterations must be positive integers.")
		}

		let options = ["--m", "--k", "--n", "--iterations", "--peak-gflops", "--bandwidth-gbps"]
		let knownArguments = Set(["benchmark", "006"] + options)
		let optionValues = valuesFollowingOptions(options, in: arguments)
		if let unknown = arguments.first(where: {
			!knownArguments.contains($0) && !optionValues.contains($0)
		}) {
			throw CLIError.invalidArguments("Unknown benchmark option '\(unknown)'.")
		}

		let lhs = try FloatTensor(
			(0..<(m * k)).map { Float(($0 % 23) - 11) / 12 },
			shape: [m, k]
		)
		let rhs = try FloatTensor(
			(0..<(k * n)).map { Float(($0 % 31) - 15) / 16 },
			shape: [k, n]
		)
		let workload = try RooflineWorkload.gemm(m: m, k: k, n: n)
		let machine = try RooflineMachine(
			peakComputeGFLOPS: peakGFLOPS,
			peakMemoryBandwidthGBps: bandwidthGBps
		)
		let model = RooflineModel.predict(workload: workload, machine: machine)
		var checksum: Float = 0

#if DEBUG
		print("Note: use 'swift run -c release inference-school benchmark 006' for meaningful timings.")
#endif
		let cpuMeasurement = try RooflineBenchmark.measure(
			iterations: iterations,
			workload: workload
		) {
			checksum += try P005GEMMSolution.multiply(lhs, rhs).storage[0]
		}
		print("CPU canonical GEMM")
		print(RooflineReport(
			workload: workload,
			assumedMachine: machine,
			model: model,
			measured: cpuMeasurement
		).rendered())

		let pipeline = try P005GEMMSolution.makeMetalPipeline()
		let metalMeasurement = try RooflineBenchmark.measure(
			iterations: iterations,
			workload: workload
		) {
			checksum += try pipeline.multiply(lhs, rhs).storage[0]
		}
		print("\nMetal canonical GEMM (end-to-end)")
		print(RooflineReport(
			workload: workload,
			assumedMachine: machine,
			model: model,
			measured: metalMeasurement
		).rendered())
		print(String(format: "Checksum: %.3f", checksum))
		print("Traffic is the algorithmic minimum; measured Metal time includes allocation, copies, submission, and synchronization.")
	}

	private static func benchmarkKVLayouts(arguments: [String]) throws {
		let layers = try integerOption("--layers", in: arguments, default: 16)
		let tokens = try integerOption("--tokens", in: arguments, default: 2_048)
		let heads = try integerOption("--heads", in: arguments, default: 8)
		let dimension = try integerOption("--dimension", in: arguments, default: 64)
		let iterations = try integerOption("--iterations", in: arguments, default: 7)
		guard layers > 0, tokens > 0, heads > 0, dimension > 0, iterations > 0 else {
			throw CLIError.invalidArguments(
				"--layers, --tokens, --heads, --dimension, and --iterations must be positive integers.")
		}
		let options = ["--layers", "--tokens", "--heads", "--dimension", "--iterations"]
		let knownArguments = Set(["benchmark", "024"] + options)
		let optionValues = valuesFollowingOptions(options, in: arguments)
		if let unknown = arguments.first(where: {
			!knownArguments.contains($0) && !optionValues.contains($0)
		}) {
			throw CLIError.invalidArguments("Unknown benchmark option '\(unknown)'.")
		}
		let configuration = try KVCacheConfiguration(
			layerCount: layers,
			keyValueHeadCount: heads,
			headDimension: dimension,
			capacity: tokens)
#if DEBUG
		print("Note: use 'swift run -c release inference-school benchmark 024' for meaningful timings.")
#endif
		print(try P024KVLayoutShootoutSolution.benchmark(
			configuration: configuration, iterations: iterations).rendered())
	}

	private static func benchmarkFusedQ4GEMV(arguments: [String]) throws {
		let outputChannels = try integerOption("--out", in: arguments, default: 1_024)
		let inputChannels = try integerOption("--in", in: arguments, default: 1_024)
		let groupSize = try integerOption("--group-size", in: arguments, default: 64)
		let iterations = try integerOption("--iterations", in: arguments, default: 20)
		guard outputChannels > 0, inputChannels > 0, groupSize > 0, iterations > 0 else {
			throw CLIError.invalidArguments(
				"--out, --in, --group-size, and --iterations must be positive integers.")
		}
		let options = ["--out", "--in", "--group-size", "--iterations"]
		let knownArguments = Set(["benchmark", "033"] + options)
		let optionValues = valuesFollowingOptions(options, in: arguments)
		if let unknown = arguments.first(where: {
			!knownArguments.contains($0) && !optionValues.contains($0)
		}) {
			throw CLIError.invalidArguments("Unknown benchmark option '\(unknown)'.")
		}

		let floatWeights = try FloatTensor(
			(0..<(outputChannels * inputChannels)).map { index in
				Float(((index * 17) % 37) - 18) / Float(3 + (index / inputChannels) % 11)
			},
			shape: [outputChannels, inputChannels])
		let input = try FloatTensor(
			(0..<inputChannels).map { Float(($0 % 23) - 11) / 12 },
			shape: [inputChannels])
		let weights = try P033FusedQ4GEMVSolution.quantize(
			floatWeights, groupSize: groupSize)
		let metalPipeline = try P033FusedQ4GEMVSolution.makeMetalPipeline()
		let floatBytes = floatWeights.elementCount * MemoryLayout<Float>.stride
		let stagedTemporaryBytes = weights.logicalValueCount * MemoryLayout<Float>.stride

#if DEBUG
		print("Note: use 'swift run -c release inference-school benchmark 033' for meaningful timings.")
#endif
		print("Canonical Q4 GEMV, weights=[\(outputChannels),\(inputChannels)], group=\(groupSize)")
		print("Float32 logical weights: \(floatBytes) bytes")
		print("Q4 packed values:       \(weights.packedValueBytes) bytes")
		print("Q4 Float32 scales:      \(weights.scaleBytes) bytes")
		print("Q4 total logical bytes: \(weights.allocatedBytes) bytes")
		print("Staged Float temporary: \(stagedTemporaryBytes) bytes per call")

		let staged = try measure(iterations: iterations) {
			try P032DequantizeThenGEMVSolution.multiply(weights, input).output.storage[0]
		}
		let fusedCPU = try measure(iterations: iterations) {
			try P033FusedQ4GEMVSolution.multiply(weights, input).output.storage[0]
		}
		let metal = try measure(iterations: iterations) {
			try metalPipeline.multiply(weights, input).output.storage[0]
		}
		printQ4Measurement(label: "CPU staged", measurement: staged)
		printQ4Measurement(label: "CPU fused", measurement: fusedCPU)
		printQ4Measurement(label: "Metal end-to-end", measurement: metal)
		print("Quantization is outside timing. Metal includes buffer allocation, copies, submission, and synchronization.")
	}

	private static func reportBufferPlans(arguments: [String]) throws {
		let tokens = try integerOption("--tokens", in: arguments, default: 128)
		let cachedTokens = try integerOption("--cached-tokens", in: arguments, default: tokens)
		guard tokens > 0, cachedTokens > 0 else {
			throw CLIError.invalidArguments("--tokens and --cached-tokens must be positive integers.")
		}
		let options = ["--tokens", "--cached-tokens"]
		let knownArguments = Set(["benchmark", "041"] + options)
		let optionValues = valuesFollowingOptions(options, in: arguments)
		if let unknown = arguments.first(where: {
			!knownArguments.contains($0) && !optionValues.contains($0)
		}) {
			throw CLIError.invalidArguments("Unknown benchmark option '\(unknown)'.")
		}
		let model = try EducationalMiniModelFixture.make()
		let comparison = try P041BufferPlanningSolution.compareDecoderPlans(
			model: model,
			prefillTokenCount: tokens,
			cachedTokenCount: cachedTokens)
		print("Deterministic first-fit arena plan for the educational mini-model")
		print("This reports requested intermediate storage; Swift tensors are not allocated from this arena.")
		printArenaPlan(label: "Prefill S=\(tokens)", plan: comparison.prefill)
		printArenaPlan(label: "Decode T=\(cachedTokens)", plan: comparison.decode)
	}

	private static func benchmarkFusedQKV(arguments: [String]) throws {
		let tokens = try integerOption("--tokens", in: arguments, default: 32)
		let iterations = try integerOption("--iterations", in: arguments, default: 20)
		guard tokens > 0, tokens <= P043FusedQKVContract.maximumSequenceLength,
			iterations > 0
		else {
			throw CLIError.invalidArguments(
				"--tokens must be in 1...\(P043FusedQKVContract.maximumSequenceLength), and --iterations must be positive.")
		}
		try validateOptions(
			arguments: arguments,
			commandPrefix: ["benchmark", "043"],
			options: ["--tokens", "--iterations"])
		let configuration = try DecoderConfiguration(
			modelDimension: 256,
			hiddenDimension: 512,
			queryHeadCount: 4,
			keyValueHeadCount: 4,
			headDimension: 64,
			rotaryDimension: 64,
			rmsNormEpsilon: 1e-5)
		func tensor(count: Int, shape: [Int], multiplier: Int) throws -> FloatTensor {
			try FloatTensor(
				(0..<count).map { Float((($0 * multiplier) % 43) - 21) / 37 },
				shape: shape)
		}
		let dimension = configuration.modelDimension
		let queryWidth = configuration.queryProjectionDimension
		let keyValueWidth = configuration.keyValueProjectionDimension
		let request = FusedQKVRequest(
			input: try tensor(count: tokens * dimension, shape: [tokens, dimension], multiplier: 7),
			gamma: try tensor(count: dimension, shape: [dimension], multiplier: 5),
			queryWeights: try tensor(count: queryWidth * dimension, shape: [queryWidth, dimension], multiplier: 11),
			keyWeights: try tensor(count: keyValueWidth * dimension, shape: [keyValueWidth, dimension], multiplier: 13),
			valueWeights: try tensor(count: keyValueWidth * dimension, shape: [keyValueWidth, dimension], multiplier: 17),
			epsilon: configuration.rmsNormEpsilon,
			configuration: configuration)
		let cost = try P043FusedQKVCostModel.compare(request)
		let pipeline = try P043FusedQKVSolution.makeMetalPipeline()
#if DEBUG
		print("Note: use 'swift run -c release inference-school benchmark 043' for meaningful timings.")
#endif
		print("Canonical fused QKV, input=[\(tokens),\(dimension)], median of \(iterations) iterations")
		print("Modeled separate: \(cost.separate.dispatchCount) dispatches, \(cost.separate.logicalTensorBytes) logical bytes, \(cost.separate.intermediateBytes) intermediate bytes")
		print("Modeled fused:    \(cost.fused.dispatchCount) dispatch, \(cost.fused.logicalTensorBytes) logical bytes, \(cost.fused.intermediateBytes) intermediate bytes")
		let separate = try measure(iterations: iterations) {
			try P043FusedQKVSolution.separate(request).queries.storage[0]
		}
		let fused = try measure(iterations: iterations) {
			try P043FusedQKVSolution.fused(request).queries.storage[0]
		}
		let metal = try measure(iterations: iterations) {
			try pipeline.run(request).result.queries.storage[0]
		}
		printQ4Measurement(label: "CPU separate", measurement: separate)
		printQ4Measurement(label: "CPU fused", measurement: fused)
		printQ4Measurement(label: "Metal end-to-end", measurement: metal)
		let execution = try pipeline.run(request)
		print("Metal accounting: \(execution.dispatchCount) dispatch, \(execution.commandBufferCount) command buffer, \(execution.hostWaitCount) host wait")
		print("Metal bytes: allocated=\(execution.allocatedBufferBytes), H2D=\(execution.hostToDeviceBytes), D2H=\(execution.deviceToHostBytes)")
		print("Metal timing includes per-call shared-buffer allocation, copies, submission, and synchronization.")
	}

	private static func profileProblem(arguments: [String]) throws {
		let problemID = try requiredProblemID(in: arguments)
		guard problemID == "044" else { throw CLIError.unknownProblem(problemID) }
		try profilePrefillDecode(arguments: arguments, command: "profile")
	}

	private static func profilePrefillDecode(arguments: [String], command: String) throws {
		let promptTokens = try integerOption("--prompt-tokens", in: arguments, default: 16)
		let trials = try integerOption("--trials", in: arguments, default: 7)
		let warmup = try integerOption("--warmup", in: arguments, default: 2)
		let decodeSteps = try integerOption("--decode-steps", in: arguments, default: 4)
		guard promptTokens > 0, trials > 0, warmup >= 0, decodeSteps > 0 else {
			throw CLIError.invalidArguments(
				"--prompt-tokens, --trials, and --decode-steps must be positive; --warmup must be nonnegative.")
		}
		try validateOptions(
			arguments: arguments,
			commandPrefix: [command, "044"],
			options: ["--prompt-tokens", "--trials", "--warmup", "--decode-steps"])
		let model = try EducationalMiniModelFixture.make()
		let base = [1, 2, 3, 4, 5, 6]
		let prompt = (0..<promptTokens).map { base[$0 % base.count] }
		let contexts = Array(Set([promptTokens, promptTokens * 4, promptTokens * 16])).sorted()
		let report = try P044PrefillDecodeProfilingSolution.profile(
			PrefillDecodeProfilingRequest(
				model: model,
				promptTokenIDs: prompt,
				decodeContextLengths: contexts,
				warmupIterations: warmup,
				measuredTrials: trials,
				decodeStepsPerTrial: decodeSteps))
#if DEBUG
		print("Note: use a release build for meaningful profile timings.")
#endif
		print("Backend: \(report.backend)")
		print("Clock: \(report.clock)")
		print("Boundary: \(report.timingBoundary)")
		print(String(
			format: "Prefill S=%d median %.3f ms p95 %.3f ms %.2f prompt tok/s",
			report.prefill.promptTokenCount,
			report.prefill.latency.medianNanoseconds / 1_000_000,
			Double(report.prefill.latency.percentileNanoseconds) / 1_000_000,
			report.prefill.promptTokensPerSecond))
		for decode in report.decode {
			print(String(
				format: "Decode T=%d median %.3f ms/token p95 %.3f ms/token %.2f tok/s",
				decode.initialContextLength,
				decode.perTokenLatency.medianNanoseconds / 1_000_000,
				Double(decode.perTokenLatency.percentileNanoseconds) / 1_000_000,
				decode.decodeTokensPerSecond))
		}
	}

	private static func runCapstone(arguments: [String]) throws {
		let prompt = try stringOption("--prompt", in: arguments, default: P047CapstoneFixture.defaultPrompt)
		let maxTokens = try integerOption("--max-tokens", in: arguments, default: 4)
		let seed = try unsignedIntegerOption("--seed", in: arguments, default: 47)
		let includeMetal = !arguments.contains("--no-metal")
		try validateOptions(
			arguments: arguments,
			commandPrefix: ["capstone"],
			options: ["--prompt", "--max-tokens", "--seed"],
			flags: ["--no-metal"])
		let request = try P047CapstoneFixture.makeRequest(
			prompt: prompt,
			maxNewTokens: maxTokens,
			seed: seed,
			includeMetalVerification: includeMetal)
		let report = try P047CapstoneSolution.run(request)
		print("Generation backend: \(report.generationBackend)")
		print("Prompt: \(report.prompt)")
		print("Prompt token IDs: \(report.promptTokenIDs)")
		print("Generated token IDs: \(report.generatedTokenIDs)")
		print("Generated bytes: \(report.generatedBytes)")
		switch report.rendering {
		case .text(let text): print("Generated text: \(text)")
		case .hexadecimal(let value): print("Generated hex: \(value)")
		}
		print("Stop reason: \(report.stopReason)")
		if let value = report.timeToFirstTokenNanoseconds {
			print(String(format: "Time to first token: %.3f ms", Double(value) / 1_000_000))
		}
		if let value = report.decodeTokensPerSecond {
			print(String(format: "Serial decode: %.2f tok/s", value))
		}
		for timing in report.timings {
			print(String(format: "  %-22s %.3f ms", (timing.name as NSString).utf8String!, Double(timing.nanoseconds) / 1_000_000))
		}
		print("KV cache counts by layer: \(report.finalCacheCounts)")
		print("Memory bytes: weights=\(report.modelWeightBytes), KV allocated=\(report.allocatedKVCacheBytes), prefill arena=\(report.prefillArenaBytes), decode arena=\(report.decodeArenaBytes)")
		print("Formats: weights=\(report.weightFormat); cache=\(report.keyValueFormat)")
		print("Optimization evidence: \(report.optimizationComparison.basis)")
		print("Rejected optimization: \(report.rejectedOptimization.name)")
		print("  \(report.rejectedOptimization.evidence)")
		print("\(report.metalVerification.label): \(report.metalVerification.status)")
		if let resources = report.metalVerification.resources {
			print("  dispatches=\(resources.dispatchCount), command buffers=\(resources.commandBufferCount), host waits=\(resources.hostWaitCount)")
			for capture in report.metalVerification.captures {
				print(String(format: "  %-30s max abs %.3e %@", (capture.name as NSString).utf8String!, capture.maximumAbsoluteError, capture.passes ? "PASS" : "FAIL"))
			}
		}
		print("Limitations:")
		for limitation in report.limitations { print("  - \(limitation)") }
	}

	private static func printArenaPlan(label: String, plan: ArenaPlan) {
		print("\n\(label)")
		print("  arena bytes:     \(plan.arenaByteCount)")
		print("  peak live bytes: \(plan.peakLiveBytes)")
		print("  naive bytes:     \(plan.naiveByteCount)")
		for placement in plan.placements {
			print("  \(placement.name): offset=\(placement.offset) bytes=\(placement.byteSize) live=\(placement.firstOperation)...\(placement.lastOperation)")
		}
		for reuse in plan.reuseAssignments where !reuse.reusesStorageFrom.isEmpty {
			print("  reuse \(reuse.buffer) <- \(reuse.reusesStorageFrom.joined(separator: ", "))")
		}
	}

	private static func measure(
		iterations: Int,
		operation: () throws -> Float
	) throws -> (nanoseconds: UInt64, checksum: Float) {
		for _ in 0..<min(iterations, 3) {
			_ = try operation()
		}

		var samples: [UInt64] = []
		var checksum: Float = 0
		for _ in 0..<iterations {
			let start = DispatchTime.now().uptimeNanoseconds
			checksum += try operation()
			samples.append(DispatchTime.now().uptimeNanoseconds - start)
		}
		samples.sort()
		return (samples[samples.count / 2], checksum)
	}

	private static func printMeasurement(
		label: String,
		measurement: (nanoseconds: UInt64, checksum: Float),
		elementCount: Int
	) {
		let seconds = Double(measurement.nanoseconds) / 1_000_000_000
		let milliseconds = seconds * 1_000
		let effectiveBytes = Double(elementCount * 2 * MemoryLayout<Float>.stride)
		let gigabytesPerSecond = effectiveBytes / seconds / 1_000_000_000
		let gigaFLOPs = Double(elementCount * 2) / seconds / 1_000_000_000
		print(
			String(
				format: "%-16s %8.3f ms  %7.2f effective GB/s  %7.2f GFLOP/s  checksum %.3f",
				(label as NSString).utf8String!,
				milliseconds,
				gigabytesPerSecond,
				gigaFLOPs,
				measurement.checksum
			)
		)
	}

	private static func printQ4Measurement(
		label: String,
		measurement: (nanoseconds: UInt64, checksum: Float)
	) {
		let milliseconds = Double(measurement.nanoseconds) / 1_000_000
		print(String(
			format: "%-18s %8.3f ms  checksum %.3f",
			(label as NSString).utf8String!, milliseconds, measurement.checksum))
	}

	private static func requiredProblemID(in arguments: [String]) throws -> String {
		guard arguments.count >= 2 else {
			throw CLIError.invalidArguments("A problem ID is required.")
		}
		guard let problem = Course.problem(id: arguments[1]), problem.isAvailable else {
			throw CLIError.unknownProblem(arguments[1])
		}
		return arguments[1]
	}

	private static func requiredProblemIDs(in arguments: [String]) throws -> [String] {
		guard arguments.count >= 2 else {
			throw CLIError.invalidArguments("A problem ID or range is required.")
		}
		let value = arguments[1]
		if let problem = Course.problem(id: value), problem.isAvailable {
			return [value]
		}
		let bounds = value.split(separator: "-", omittingEmptySubsequences: false)
		guard bounds.count == 2,
			let first = Int(bounds[0]),
			let last = Int(bounds[1]),
			first <= last
		else { throw CLIError.unknownProblem(value) }
		var problemIDs: [String] = []
		for number in first...last {
			let problemID = String(format: "%03d", number)
			guard let problem = Course.problem(id: problemID), problem.isAvailable else {
				throw CLIError.unknownProblem(problemID)
			}
			problemIDs.append(problemID)
		}
		return problemIDs
	}

	private static func integerOption(
		_ option: String,
		in arguments: [String],
		default defaultValue: Int
	) throws -> Int {
		guard let optionIndex = arguments.firstIndex(of: option) else {
			return defaultValue
		}
		let valueIndex = arguments.index(after: optionIndex)
		guard valueIndex < arguments.endIndex, let value = Int(arguments[valueIndex]) else {
			throw CLIError.invalidArguments("\(option) requires an integer value.")
		}
		return value
	}

	private static func doubleOption(
		_ option: String,
		in arguments: [String],
		default defaultValue: Double
	) throws -> Double {
		guard let optionIndex = arguments.firstIndex(of: option) else {
			return defaultValue
		}
		let valueIndex = arguments.index(after: optionIndex)
		guard valueIndex < arguments.endIndex, let value = Double(arguments[valueIndex]) else {
			throw CLIError.invalidArguments("\(option) requires a numeric value.")
		}
		return value
	}

	private static func valuesFollowingOptions(
		_ options: [String],
		in arguments: [String]
	) -> Set<String> {
		var values: Set<String> = []
		for option in options {
			if let index = arguments.firstIndex(of: option) {
				let valueIndex = arguments.index(after: index)
				if valueIndex < arguments.endIndex {
					values.insert(arguments[valueIndex])
				}
			}
		}
		return values
	}

	private static func stringOption(
		_ option: String,
		in arguments: [String],
		default defaultValue: String
	) throws -> String {
		guard let optionIndex = arguments.firstIndex(of: option) else { return defaultValue }
		let valueIndex = arguments.index(after: optionIndex)
		guard valueIndex < arguments.endIndex else {
			throw CLIError.invalidArguments("\(option) requires a value.")
		}
		return arguments[valueIndex]
	}

	private static func unsignedIntegerOption(
		_ option: String,
		in arguments: [String],
		default defaultValue: UInt64
	) throws -> UInt64 {
		guard let optionIndex = arguments.firstIndex(of: option) else { return defaultValue }
		let valueIndex = arguments.index(after: optionIndex)
		guard valueIndex < arguments.endIndex, let value = UInt64(arguments[valueIndex]) else {
			throw CLIError.invalidArguments("\(option) requires a nonnegative integer value.")
		}
		return value
	}

	private static func validateOptions(
		arguments: [String],
		commandPrefix: [String],
		options: [String],
		flags: [String] = []
	) throws {
		let knownArguments = Set(commandPrefix + options + flags)
		let optionValues = valuesFollowingOptions(options, in: arguments)
		if let unknown = arguments.first(where: {
			!knownArguments.contains($0) && !optionValues.contains($0)
		}) {
			throw CLIError.invalidArguments("Unknown option '\(unknown)'.")
		}
	}

}

public enum RuntimeRegistry {
	public static func runner(for problemID: String) -> ProblemRunner? {
		switch problemID {
		case "001":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P001VectorDotExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P001VectorDot.metal",
				cpuCheck: { useSolution in
					P001VectorDotJudge.evaluate(
						useSolution ? P001VectorDotSolution.dot : P001VectorDotExercise.dot
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P001VectorDotSolution.makeMetalPipeline()
						: P001VectorDotExercise.makeMetalPipeline()
					return P001VectorDotJudge.evaluate(pipeline.dot)
				}
			)
		case "002":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P002TensorStorageExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P002TensorStorageJudge.evaluate(
						useSolution ? P002TensorStorageSolution.gather : P002TensorStorageExercise.gather
					)
				},
				metalCheck: nil
			)
		case "003":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P003TransposeExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P003Transpose.metal",
				cpuCheck: { useSolution in
					P003TransposeJudge.evaluate(
						useSolution ? P003TransposeSolution.transpose : P003TransposeExercise.transpose
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P003TransposeSolution.makeMetalPipeline()
						: P003TransposeExercise.makeMetalPipeline()
					return P003TransposeJudge.evaluate(pipeline.transpose)
				}
			)
		case "004":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P004GEMVExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P004GEMV.metal",
				cpuCheck: { useSolution in
					P004GEMVJudge.evaluate(
						useSolution ? P004GEMVSolution.multiply : P004GEMVExercise.multiply
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P004GEMVSolution.makeMetalPipeline()
						: P004GEMVExercise.makeMetalPipeline()
					return P004GEMVJudge.evaluate(pipeline.multiply)
				}
			)
		case "005":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P005GEMMExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P005GEMM.metal",
				cpuCheck: { useSolution in
					P005GEMMJudge.evaluate(
						useSolution ? P005GEMMSolution.multiply : P005GEMMExercise.multiply
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P005GEMMSolution.makeMetalPipeline()
						: P005GEMMExercise.makeMetalPipeline()
					return P005GEMMJudge.evaluate(pipeline.multiply)
				}
			)
		case "006":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P006RooflineExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P006RooflineJudge.evaluate(
						useSolution ? P006RooflineSolution.predict : P006RooflineExercise.predict
					)
				},
				metalCheck: nil
			)
		case "007":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P007ActivationExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P007Activation.metal",
				cpuCheck: { useSolution in
					P007ActivationJudge.evaluate(
						useSolution ? P007ActivationSolution.apply : P007ActivationExercise.apply
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P007ActivationSolution.makeMetalPipeline()
						: P007ActivationExercise.makeMetalPipeline()
					return P007ActivationJudge.evaluate(pipeline.apply)
				}
			)
		case "008":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P008SwiGLUExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P008SwiGLU.metal",
				cpuCheck: { useSolution in
					P008SwiGLUJudge.evaluate(
						useSolution ? P008SwiGLUSolution.apply : P008SwiGLUExercise.apply
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P008SwiGLUSolution.makeMetalGatePipeline()
						: P008SwiGLUExercise.makeMetalGatePipeline()
					return P008SwiGLUJudge.evaluateGate(pipeline.apply)
				}
			)
		case "009":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P009SoftmaxExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P009Softmax.metal",
				cpuCheck: { useSolution in
					P009SoftmaxJudge.evaluate(
						useSolution ? P009SoftmaxSolution.apply : P009SoftmaxExercise.apply
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P009SoftmaxSolution.makeMetalPipeline()
						: P009SoftmaxExercise.makeMetalPipeline()
					return P009SoftmaxJudge.evaluate(pipeline.apply)
				}
			)
		case "010":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P010RMSNormExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P010RMSNorm.metal",
				cpuCheck: { useSolution in
					P010RMSNormJudge.evaluate(
						useSolution ? P010RMSNormSolution.apply : P010RMSNormExercise.apply
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P010RMSNormSolution.makeMetalPipeline()
						: P010RMSNormExercise.makeMetalPipeline()
					return P010RMSNormJudge.evaluate(pipeline.apply)
				}
			)
		case "011":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P011ResidualPrecisionExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P011ResidualPrecisionJudge.evaluate(
						useSolution ? P011ResidualPrecisionSolution.accumulate : P011ResidualPrecisionExercise.accumulate
					)
				},
				metalCheck: nil
			)
		case "012":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P012FusedRMSNormGEMVExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P012FusedRMSNormGEMV.metal",
				cpuCheck: { useSolution in
					P012FusedRMSNormGEMVJudge.evaluate(
						useSolution ? P012FusedRMSNormGEMVSolution.baseline : P012FusedRMSNormGEMVExercise.project
					)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P012FusedRMSNormGEMVSolution.makeFusedMetalPipeline()
						: P012FusedRMSNormGEMVExercise.makeFusedMetalPipeline()
					return P012FusedRMSNormGEMVJudge.evaluate(pipeline.project)
				}
			)
		case "013":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P013EmbeddingExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P013Embedding.metal",
				cpuCheck: { useSolution in
					P013EmbeddingJudge.evaluate(useSolution ? P013EmbeddingSolution.apply : P013EmbeddingExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P013EmbeddingSolution.makeMetalPipeline() : P013EmbeddingExercise.makeMetalPipeline()
					return P013EmbeddingJudge.evaluate(pipeline.apply)
				}
			)
		case "014":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P014QKVProjectionExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P014QKVProjectionJudge.evaluate(useSolution ? P014QKVProjectionSolution.project : P014QKVProjectionExercise.project)
				},
				metalCheck: nil
			)
		case "015":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P015RoPEExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P015RoPE.metal",
				cpuCheck: { useSolution in
					P015RoPEJudge.evaluate(useSolution ? P015RoPESolution.apply : P015RoPEExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P015RoPESolution.makeMetalPipeline() : P015RoPEExercise.makeMetalPipeline()
					return P015RoPEJudge.evaluate { try pipeline.apply($0, $1, rotaryDimension: $2, base: $3, queryPositionOffset: $4, keyPositionOffset: $5) }
				}
			)
		case "016":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P016CausalAttentionExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P016CausalAttention.metal",
				cpuCheck: { useSolution in
					P016CausalAttentionJudge.evaluate(useSolution ? P016CausalAttentionSolution.apply : P016CausalAttentionExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P016CausalAttentionSolution.makeMetalPipeline() : P016CausalAttentionExercise.makeMetalPipeline()
					return P016CausalAttentionJudge.evaluate { try pipeline.apply($0, $1, $2, configuration: $3) }
				}
			)
		case "017":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P017MultiHeadAttentionExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P017MultiHeadAttention.metal",
				cpuCheck: { useSolution in
					P017MultiHeadAttentionJudge.evaluate(useSolution ? P017MultiHeadAttentionSolution.apply : P017MultiHeadAttentionExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P017MultiHeadAttentionSolution.makeMetalPipeline() : P017MultiHeadAttentionExercise.makeMetalPipeline()
					return P017MultiHeadAttentionJudge.evaluate { try pipeline.apply($0, $1, $2, configuration: $3) }
				}
			)
		case "018":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P018GroupedQueryAttentionExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P018GroupedQueryAttention.metal",
				cpuCheck: { useSolution in
					P018GroupedQueryAttentionJudge.evaluate(useSolution ? P018GroupedQueryAttentionSolution.apply : P018GroupedQueryAttentionExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P018GroupedQueryAttentionSolution.makeMetalPipeline() : P018GroupedQueryAttentionExercise.makeMetalPipeline()
					return P018GroupedQueryAttentionJudge.evaluate { try pipeline.apply($0, $1, $2, configuration: $3) }
				}
			)
		case "019":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P019OnlineAttentionExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P019OnlineAttention.metal",
				cpuCheck: { useSolution in
					P019OnlineAttentionJudge.evaluate(useSolution ? P019OnlineAttentionSolution.apply : P019OnlineAttentionExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P019OnlineAttentionSolution.makeMetalPipeline() : P019OnlineAttentionExercise.makeMetalPipeline()
					return P019OnlineAttentionJudge.evaluate { try pipeline.apply($0, $1, $2, configuration: $3) }
				}
			)
		case "020":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P020TiledAttentionExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P020TiledAttention.metal",
				cpuCheck: { useSolution in
					P020TiledAttentionJudge.evaluate(useSolution ? P020TiledAttentionSolution.apply : P020TiledAttentionExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P020TiledAttentionSolution.makeMetalPipeline() : P020TiledAttentionExercise.makeMetalPipeline()
					return P020TiledAttentionJudge.evaluate { try pipeline.apply($0, $1, $2, configuration: $3) }
				}
			)
		case "021":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P021SlidingWindowAttentionExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P021SlidingWindowAttention.metal",
				cpuCheck: { useSolution in
					P021SlidingWindowAttentionJudge.evaluate(useSolution ? P021SlidingWindowAttentionSolution.apply : P021SlidingWindowAttentionExercise.apply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution ? P021SlidingWindowAttentionSolution.makeMetalPipeline() : P021SlidingWindowAttentionExercise.makeMetalPipeline()
					return P021SlidingWindowAttentionJudge.evaluate { try pipeline.apply($0, $1, $2, configuration: $3, window: $4) }
				}
			)
		case "022":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P022ContiguousKVCacheExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P022ContiguousKVCacheJudge.evaluate(
						useSolution ? P022ContiguousKVCacheSolution.run : P022ContiguousKVCacheExercise.run)
				},
				metalCheck: nil
			)
		case "023":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P023CachedAttentionExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P023CachedAttention.metal",
				cpuCheck: { useSolution in
					P023CachedAttentionJudge.evaluate(
						useSolution ? P023CachedAttentionSolution.run : P023CachedAttentionExercise.run)
				},
				metalCheck: { useSolution in
					if useSolution {
						let pipeline = try P023CachedAttentionSolution.makeMetalPipeline()
						return P023CachedAttentionJudge.evaluate {
							try P023CachedAttentionSolution.runMetal($0, pipeline: pipeline)
						}
					}
					let pipeline = try P023CachedAttentionExercise.makeMetalPipeline()
					return P023CachedAttentionJudge.evaluate {
						try P023CachedAttentionExercise.runMetal($0, pipeline: pipeline)
					}
				}
			)
		case "024":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P024KVLayoutShootoutExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P024KVLayoutShootoutJudge.evaluate(
						useSolution ? P024KVLayoutShootoutSolution.run : P024KVLayoutShootoutExercise.run)
				},
				metalCheck: nil
			)
		case "025":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P025SharedKVHeadsExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P025SharedKVHeadsJudge.evaluate(
						useSolution ? P025SharedKVHeadsSolution.run : P025SharedKVHeadsExercise.run)
				},
				metalCheck: nil
			)
		case "026":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P026RingKVCacheExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P026RingKVCacheJudge.evaluate(
						useSolution ? P026RingKVCacheSolution.run : P026RingKVCacheExercise.run)
				},
				metalCheck: nil
			)
		case "027":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P027PagedKVCacheExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P027PagedKVCacheJudge.evaluate(
						useSolution ? P027PagedKVCacheSolution.run : P027PagedKVCacheExercise.run)
				},
				metalCheck: nil
			)
		case "028":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P028QuantizedKVCacheExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P028QuantizedKVCache.metal",
				cpuCheck: { useSolution in
					P028QuantizedKVCacheJudge.evaluate(
						useSolution ? P028QuantizedKVCacheSolution.run : P028QuantizedKVCacheExercise.run)
				},
				metalCheck: { useSolution in
					if useSolution {
						let pipeline = try P028QuantizedKVCacheSolution.makeMetalPipeline()
						return P028QuantizedKVCacheJudge.evaluate {
							try P028QuantizedKVCacheSolution.runMetal($0, pipeline: pipeline)
						}
					}
					let pipeline = try P028QuantizedKVCacheExercise.makeMetalPipeline()
					return P028QuantizedKVCacheJudge.evaluate {
						try P028QuantizedKVCacheExercise.runMetal($0, pipeline: pipeline)
					}
				}
			)
		case "029":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P029SymmetricInt8Exercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P029SymmetricInt8Judge.evaluate(
						useSolution ? P029SymmetricInt8Solution.quantize : P029SymmetricInt8Exercise.quantize)
				},
				metalCheck: nil)
		case "030":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P030GroupwiseInt8Exercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P030GroupwiseInt8Judge.evaluate(
						useSolution ? P030GroupwiseInt8Solution.compare : P030GroupwiseInt8Exercise.compare)
				},
				metalCheck: nil)
		case "031":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P031PackQ4Exercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P031PackQ4Judge.evaluate(
						useSolution ? P031PackQ4Solution.pack : P031PackQ4Exercise.pack)
				},
				metalCheck: nil)
		case "032":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P032DequantizeThenGEMVExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P032DequantizeThenGEMVJudge.evaluate(
						useSolution ? P032DequantizeThenGEMVSolution.multiply : P032DequantizeThenGEMVExercise.multiply)
				},
				metalCheck: nil)
		case "033":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P033FusedQ4GEMVExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P033FusedQ4GEMV.metal",
				cpuCheck: { useSolution in
					P033FusedQ4GEMVJudge.evaluate(
						useSolution ? P033FusedQ4GEMVSolution.multiply : P033FusedQ4GEMVExercise.multiply)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P033FusedQ4GEMVSolution.makeMetalPipeline()
						: P033FusedQ4GEMVExercise.makeMetalPipeline()
					return P033FusedQ4GEMVJudge.evaluate(pipeline.multiply)
				})
		case "034":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P034QuantizationPropagationExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P034QuantizationPropagationJudge.evaluate(
						useSolution ? P034QuantizationPropagationSolution.investigate : P034QuantizationPropagationExercise.investigate)
				},
				metalCheck: nil)
		case "035":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P035DecoderBlockExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P035DecoderBlockJudge.evaluate(
						useSolution ? P035DecoderBlockSolution.apply : P035DecoderBlockExercise.apply)
				},
				metalCheck: nil)
		case "036":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P036WeightContainerExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P036WeightContainerJudge.evaluate(
						useSolution ? P036WeightContainerSolution.parse : P036WeightContainerExercise.parse)
				},
				metalCheck: nil)
		case "037":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P037ByteBPEExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P037ByteBPEJudge.evaluate(
						encode: useSolution ? P037ByteBPESolution.encode : P037ByteBPEExercise.encode,
						decodeBytes: useSolution ? P037ByteBPESolution.decodeBytes : P037ByteBPEExercise.decodeBytes,
						decodeText: useSolution ? P037ByteBPESolution.decodeText : P037ByteBPEExercise.decodeText)
				},
				metalCheck: nil)
		case "038":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P038LogitsSamplingExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P038LogitsSamplingJudge.evaluate(
						useSolution ? P038LogitsSamplingSolution.sample : P038LogitsSamplingExercise.sample)
				},
				metalCheck: nil)
		case "039":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P039PromptPrefillExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P039PromptPrefillJudge.evaluate(
						useSolution ? P039PromptPrefillSolution.run : P039PromptPrefillExercise.run)
				},
				metalCheck: nil)
		case "040":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P040AutoregressiveDecodeExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P040AutoregressiveDecodeJudge.evaluate(
						useSolution ? P040AutoregressiveDecodeSolution.run : P040AutoregressiveDecodeExercise.run)
				},
				metalCheck: nil)
		case "041":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P041BufferPlanningExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P041BufferPlanningJudge.evaluate(
						useSolution ? P041BufferPlanningSolution.plan : P041BufferPlanningExercise.plan)
				},
				metalCheck: nil)
		case "042":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P042CheckpointParityExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P042CheckpointParityJudge.evaluate(
						useSolution ? P042CheckpointParitySolution.compare : P042CheckpointParityExercise.compare)
				},
				metalCheck: nil)
		case "043":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P043FusedQKVExercise.swift"],
				metalFile: "Sources/InferenceSchoolExercises/Metal/P043FusedQKV.metal",
				cpuCheck: { useSolution in
					P043FusedQKVJudge.evaluate(
						useSolution ? P043FusedQKVSolution.fused : P043FusedQKVExercise.fused)
				},
				metalCheck: { useSolution in
					let pipeline = try useSolution
						? P043FusedQKVSolution.makeMetalPipeline()
						: P043FusedQKVExercise.makeMetalPipeline()
					return P043FusedQKVJudge.evaluate(pipeline.project)
				})
		case "044":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P044PrefillDecodeProfilingExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P044ProfilingJudge.evaluate(
						useSolution ? P044PrefillDecodeProfilingSolution.profile : P044PrefillDecodeProfilingExercise.profile)
				},
				metalCheck: nil)
		case "045":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P045BatchSchedulingExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P045BatchSchedulingJudge.evaluate(
						useSolution ? P045BatchSchedulingSolution.simulate : P045BatchSchedulingExercise.simulate)
				},
				metalCheck: nil)
		case "046":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P046SpeculativeDecodingExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P046SpeculativeDecodingJudge.evaluate(
						useSolution ? P046SpeculativeDecodingSolution.decodeBlock : P046SpeculativeDecodingExercise.decodeBlock)
				},
				metalCheck: nil)
		case "047":
			ProblemRunner(
				exerciseFiles: ["Sources/InferenceSchoolExercises/P047CapstoneExercise.swift"],
				metalFile: nil,
				cpuCheck: { useSolution in
					P047CapstoneJudge.evaluate(
						useSolution ? P047CapstoneSolution.run : P047CapstoneExercise.run)
				},
				metalCheck: nil)
		default:
			nil
		}
	}

}

private extension InferenceSchoolCommand {
	private static func printUsage() {
		print(
			"""
			\(Course.title) - learn LLM inference by building it

			Usage:
			  inference-school list
			  inference-school learn <problem-id>
			  inference-school show <problem-id>
			  inference-school check <problem-id | start-end> [--cpu | --metal] [--solution]
			                 [--format text | jsonl]
			  inference-school benchmark 001 [--size N] [--iterations N]
			  inference-school benchmark 006 [--m M] [--k K] [--n N] [--iterations N]
			                         [--peak-gflops N] [--bandwidth-gbps N]
			  inference-school benchmark 024 [--layers L] [--tokens T] [--heads H]
			                         [--dimension D] [--iterations N]
			  inference-school benchmark 033 [--out O] [--in I] [--group-size G]
			                         [--iterations N]
			  inference-school benchmark 041 [--tokens S] [--cached-tokens T]
			  inference-school benchmark 043 [--tokens S] [--iterations N]
			  inference-school profile 044 [--prompt-tokens S] [--trials N] [--warmup N]
			                      [--decode-steps N]
			  inference-school benchmark 044 [same options as profile 044]
			  inference-school capstone [--prompt TEXT] [--max-tokens N] [--seed N] [--no-metal]

			Start with:
			  swift run inference-school learn 001
			"""
		)
	}
}