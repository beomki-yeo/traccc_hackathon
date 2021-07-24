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
namespace detail {

template <typename surface_t>
static __CUDA_HOST_DEVICE__ Acts::BoundVector
transform_free_to_bound_parameters(const Acts::FreeVector& free_vector,
                                   const surface_t& surface) {

    Acts::BoundVector result;

    // global -> local position
    const auto& global = free_vector.template segment<3>(Acts::eFreePos0);
    auto local = surface.global_to_local(global);
    result.template segment<2>(Acts::eBoundLoc0) = local.template segment<2>(0);

    // dir transform
    const auto& dir = free_vector.template segment<3>(Acts::eFreeDir0);
    double phi = std::atan2(dir[1], dir[0]);
    double theta =
        std::atan2(std::sqrt(dir[0] * dir[0] + dir[1] * dir[1]), dir[2]);

    result[Acts::eBoundPhi] = phi;
    result[Acts::eBoundTheta] = theta;

    // time
    result[Acts::eBoundTime] = free_vector[Acts::eFreeTime];

    // q over p
    result[Acts::eBoundQOverP] = free_vector[Acts::eFreeQOverP];

    return result;
}

}  // namespace detail
}  // namespace traccc
