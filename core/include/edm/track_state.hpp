/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */


#pragma once

#include "edm/collection.hpp"

namespace traccc{

template < typename measurement_t, typename parameters_t >
struct track_state{

    parameters_t m_predicted;        
    parameters_t m_filtered;
    parameters_t m_smoothed;

    using jacobian_t = typename parameters_t::covariance_t;
    jacobian_t m_jacobian;

    measurement_t m_measurement;

    using projector_t = typename measurement_t::projector_t;    
    projector_t m_projector;
    
    __CUDA_HOST_DEVICE__
    auto& predicted() { return m_predicted; }

    __CUDA_HOST_DEVICE__
    auto& filtered() { return m_filtered; }

    __CUDA_HOST_DEVICE__
    auto& smoothed() { return m_smoothed; }

    __CUDA_HOST_DEVICE__
    auto& jacobian() { return m_jacobian; }

    __CUDA_HOST_DEVICE__
    auto& measurement() { return m_measurement; }

    __CUDA_HOST_DEVICE__
    auto& projector() { return m_projector; }

    ////
    projector_t m_projector2;
    __CUDA_HOST_DEVICE__
    auto& projector2() { return m_projector2; }    
};
    

/// Convenience declaration for the surface collection type to use in host
/// code
template < typename measurement_t, typename parameters_t >
using host_track_state_collection = host_collection< track_state< measurement_t,
parameters_t> >;

/// Convenience declaration for the surface collection type to use in device
/// code
template < typename measurement_t, typename parameters_t >
using device_track_state_collection = device_collection< track_state<
measurement_t, parameters_t> >;

template < typename measurement_t, typename parameters_t >
using track_state_collection_data = collection_data< track_state< measurement_t,
parameters_t> >;

template < typename measurement_t, typename parameters_t >
using track_state_collection_buffer = collection_buffer< track_state< measurement_t,
parameters_t> >;


template < typename measurement_t, typename parameters_t >
using track_state_collection_view = collection_view< track_state< measurement_t,
parameters_t> >;
    
} // namespace traccc

