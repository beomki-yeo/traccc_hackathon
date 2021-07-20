/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/collection.hpp"
#include "edm/track_parameters.hpp"

// Acts
#include "Acts/Definitions/TrackParametrization.hpp"

namespace traccc {

struct truth_particle {
    particle_id pid;
    int p_type;
    Acts::ActsScalar mass;
    free_track_parameters vertex;
};

/// Convenience declaration for the truth_particle collection type to use in
/// host code
using host_truth_particle_collection = host_collection<truth_particle>;

/// Convenience declaration for the truth_particle collection type to use in
/// device code
using device_truth_particle_collection = device_collection<truth_particle>;

using truth_particle_collection_data = collection_data<truth_particle>;

using truth_particle_collection_buffer = collection_buffer<truth_particle>;

using truth_particle_collection_view = collection_view<truth_particle>;

}  // namespace traccc
