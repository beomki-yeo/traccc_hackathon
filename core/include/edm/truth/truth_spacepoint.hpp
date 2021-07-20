/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/collection.hpp"
#include "edm/spacepoint.hpp"
#include "edm/truth/truth_particle.hpp"

// Acts
#include "Acts/Definitions/TrackParametrization.hpp"

namespace traccc {

using host_truth_spacepoint_collection = host_collection<spacepoint>;

using device_truth_spacepoint_collection = device_collection<spacepoint>;

/// Convenience declaration for the spacepoint container type to use in host
/// code
using host_truth_spacepoint_container =
    host_container<truth_particle, spacepoint>;

/// Convenience declaration for the spacepoint container type to use in device
/// code
using device_truth_spacepoint_container =
    device_container<truth_particle, spacepoint>;
/// Convenience declaration for the spacepoint container data type to use in
/// host code
using truth_spacepoint_container_data =
    container_data<truth_particle, spacepoint>;

/// Convenience declaration for the spacepoint container buffer type to use in
/// host code
using truth_spacepoint_container_buffer =
    container_buffer<truth_particle, spacepoint>;

/// Convenience declaration for the spacepoint container view type to use in
/// host code
using truth_spacepoint_container_view =
    container_view<truth_particle, spacepoint>;

}  // namespace traccc
