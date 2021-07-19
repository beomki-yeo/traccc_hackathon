/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <edm/track_parameters.hpp>

// Acts
#include "Acts/EventData/TrackParameters.hpp"

namespace traccc {

std::tuple<bound_track_parameters, typename bound_track_parameters::jacobian_t,
           double>
bound_state() {}

}  // namespace traccc
