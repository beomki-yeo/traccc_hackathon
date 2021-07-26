/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/measurement.hpp"
#include "edm/truth/truth_particle.hpp"

// Acts
#include "Acts/Definitions/TrackParametrization.hpp"

namespace traccc {

using host_truth_measurement_collection = host_collection<measurement>;

using device_truth_measurement_collection = device_collection<measurement>;

/// Convenience declaration for the measurement container type to use in host
/// code
using host_truth_measurement_container =
    host_container<truth_particle, measurement>;

/// Convenience declaration for the measurement container type to use in device
/// code
using device_truth_measurement_container =
    device_container<truth_particle, measurement>;
/// Convenience declaration for the measurement container data type to use in
/// host code
using truth_measurement_container_data =
    container_data<truth_particle, measurement>;

/// Convenience declaration for the measurement container buffer type to use in
/// host code
using truth_measurement_container_buffer =
    container_buffer<truth_particle, measurement>;

/// Convenience declaration for the measurement container view type to use in
/// host code
using truth_measurement_container_view =
    container_view<truth_particle, measurement>;

}  // namespace traccc
