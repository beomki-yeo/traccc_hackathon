/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "edm/collection.hpp"
#include "utils/arch_qualifiers.hpp"

// Acts
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"

namespace traccc {

struct bound_track_parameters{
    using vector_t = Acts::BoundVector;
    using covariance_t = Acts::BoundSymMatrix;
    
    vector_t m_vector;
    covariance_t m_covariance;

    __CUDA_HOST_DEVICE__
    auto& vector() { return m_vector; }

    __CUDA_HOST_DEVICE__
    auto& covariance() { return m_covariance; }    
};

using host_bound_track_parameters_collection =
    host_collection<bound_track_parameters>;

using device_bound_track_parameters_collection =
    device_collection<bound_track_parameters>;

    
struct free_track_parameters{
    Acts::FreeVector m_params;
    Acts::FreeSymMatrix m_cov;
};
        
using host_free_track_parameters_collection =
    host_collection<free_track_parameters>;

using device_free_track_parameters_collection =
    device_collection<free_track_parameters>;


    
    /*    
/// Convenience declaration for the track_parameters collection type to use
/// in host code
using host_track_parameters_collection =
    track_parameters_collection<vecmem::vector>;

/// Convenience declaration for the track_parameters collection type to use
/// in device code
using device_track_parameters_collection =
    track_parameters_collection<vecmem::device_vector>;

/// Convenience declaration for the track_parameters container type to use in
/// host code
using host_track_parameters_container =
    host_container<unsigned int, bound_parameters>;

/// Convenience declaration for the track_parameters container type to use in
/// device code
using device_track_parameters_container =
    device_container<unsigned int, bound_parameters>;

/// Convenience declaration for the track_parameters container data type to
/// use in host code
using track_parameters_container_data =
    container_data<unsigned int, bound_parameters>;

/// Convenience declaration for the track_parameters container buffer type to
/// use in host code
using track_parameters_container_buffer =
    container_buffer<unsigned int, bound_parameters>;

/// Convenience declaration for the track_parameters container view type to
/// use in host code
using track_parameters_container_view =
    container_view<unsigned int, bound_parameters>;
    */
};  // namespace traccc
