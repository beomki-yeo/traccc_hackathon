/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"

// traccc
#include "edm/track_parameters.hpp"
#include "geometry/surface.hpp"
#include "utils/unit_vectors.hpp"

namespace traccc {
namespace detail{
    
template< typename surface_t >
static __CUDA_HOST_DEVICE__
Acts::FreeVector transform_bound_to_free_parameters(const Acts::BoundVector& bound_vector, const surface_t& surface){
    Acts::FreeVector result;

    // position transform
    Acts::Vector2 local(bound_vector[Acts::eBoundLoc0], bound_vector[Acts::eBoundLoc1]);    
    result.template segment<3>(Acts::eFreePos0) = surface.local_to_global(local);
    
    // dir transform
    Acts::Vector3 dir = make_direction_unit_from_phi_theta(bound_vector[Acts::eBoundPhi], bound_vector[Acts::eBoundTheta]);
    result.template segment<3>(Acts::eFreeDir0) = dir;
    
    // time
    result[Acts::eFreeTime] = bound_vector[Acts::eBoundTime];

    // q over p
    result[Acts::eFreeQOverP] = bound_vector[Acts::eBoundQOverP];
    
    return result;    
}

} // namespace detail    
} // namespace traccc
