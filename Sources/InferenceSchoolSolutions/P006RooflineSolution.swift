import InferenceSchoolCore

public enum P006RooflineSolution {
    public static func predict(
        workload: RooflineWorkload,
        machine: RooflineMachine
    ) -> RooflinePrediction {
        RooflineModel.predict(workload: workload, machine: machine)
    }
}