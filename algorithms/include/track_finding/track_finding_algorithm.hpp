#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "Acts/TrackFinding/MeasurementSelector.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/BareAlgorithm.hpp"
#include "ActsExamples/MagneticField/MagneticField.hpp"


namespace traccc{

class track_finding_algorithm {

public:

    track_finding_algorithm(){}
    
// acts_cpu
    using TrackFinderOptions =
	Acts::CombinatorialKalmanFilterOptions<ActsExamples::MeasurementCalibrator,
					       Acts::MeasurementSelector>;

    using TrackFinderResult = std::vector<
	Acts::Result<Acts::CombinatorialKalmanFilterResult<ActsExamples::IndexSourceLink>>>;
    using TrackFinderFunction = std::function<TrackFinderResult(
	const ActsExamples::IndexSourceLinkContainer&,
	const ActsExamples::TrackParametersContainer&,
	const TrackFinderOptions&)>;

private:
    
};

}
