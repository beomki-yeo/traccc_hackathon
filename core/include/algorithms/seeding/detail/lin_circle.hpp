/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once
#include <algorithms/seeding/detail/singlet.hpp>

namespace traccc{

struct lin_circle {
    
    float m_Zo;
    float m_cotTheta;
    float m_iDeltaR;
    float m_Er;
    float m_U;
    float m_V;
    
    __CUDA_HOST_DEVICE__
    const float& Zo() const { return m_Zo; }

    __CUDA_HOST_DEVICE__
    const float& cotTheta() const { return m_cotTheta; }

    __CUDA_HOST_DEVICE__
    const float& iDeltaR() const { return m_iDeltaR; }

    __CUDA_HOST_DEVICE__
    const float& Er() const { return m_Er; }

    __CUDA_HOST_DEVICE__
    const float& U() const { return m_U; }
    
    __CUDA_HOST_DEVICE__
    const float& V() const { return m_V; }

};



    
    /// Container of lin_circle belonging to one detector module
    template< template< typename > class vector_t >
    using lin_circle_collection = vector_t< lin_circle >;

    /// Convenience declaration for the lin_circle collection type to use in host code
    using host_lin_circle_collection
    = lin_circle_collection< vecmem::vector >;

    /// Convenience declaration for the lin_circle collection type to use in device code
    using device_lin_circle_collection
    = lin_circle_collection< vecmem::device_vector >;

    /// Convenience declaration for the lin_circle container type to use in host code
    using host_lin_circle_container
    = host_container< int, lin_circle >;

    /// Convenience declaration for the lin_circle container type to use in device code
    using device_lin_circle_container
    = device_container< int, lin_circle>;

    /// Convenience declaration for the lin_circle container data type to use in host code
    using lin_circle_container_data
    = container_data< int, lin_circle >;

    /// Convenience declaration for the lin_circle container buffer type to use in host code
    using lin_circle_container_buffer
    = container_buffer< int, lin_circle >;

    /// Convenience declaration for the lin_circle container view type to use in host code
    using lin_circle_container_view
    = container_view< int, lin_circle >;    
    
}// namespace traccc

    
