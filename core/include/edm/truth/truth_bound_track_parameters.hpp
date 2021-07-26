/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/collection.hpp"
#include "edm/track_parameters.hpp"
#include "edm/truth/truth_particle.hpp"

// Acts
#include "Acts/Definitions/TrackParametrization.hpp"

namespace traccc {

using host_truth_bound_track_parameters_collection =
    host_collection<bound_track_parameters>;

using device_truth_bound_track_parameters_collection =
    device_collection<bound_track_parameters>;

/// Convenience declaration for the bound_track_parameters container type to use
/// in host code
using host_truth_bound_track_parameters_container =
    host_container<truth_particle, bound_track_parameters>;

/// Convenience declaration for the bound_track_parameters container type to use
/// in device code
using device_truth_bound_track_parameters_container =
    device_container<truth_particle, bound_track_parameters>;
/// Convenience declaration for the bound_track_parameters container data type
/// to use in host code
using truth_bound_track_parameters_container_data =
    container_data<truth_particle, bound_track_parameters>;

/// Convenience declaration for the bound_track_parameters container buffer type
/// to use in host code
using truth_bound_track_parameters_container_buffer =
    container_buffer<truth_particle, bound_track_parameters>;

/// Convenience declaration for the bound_track_parameters container view type
/// to use in host code
using truth_bound_track_parameters_container_view =
    container_view<truth_particle, bound_track_parameters>;

}  // namespace traccc
