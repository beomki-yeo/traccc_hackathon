/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

namespace traccc{    
namespace cuda{

    /// Container of compatseed belonging to one detector module
    template< template< typename > class vector_t >
    using compatseed_collection = vector_t< float >;
    
    /// Convenience declaration for the compatseed collection type to use in host code
    using host_compatseed_collection
    = compatseed_collection< vecmem::vector >;

    /// Convenience declaration for the compatseed collection type to use in device code
    using device_compatseed_collection
    = compatseed_collection< vecmem::device_vector >;

    /// Convenience declaration for the compatseed container type to use in host code
    using host_compatseed_container
    = host_container< int, float >;

    /// Convenience declaration for the compatseed container type to use in device code
    using device_compatseed_container
    = device_container< int, float >;

    /// Convenience declaration for the compatseed container data type to use in host code
    using compatseed_container_data
    = container_data< int, float >;

    /// Convenience declaration for the compatseed container buffer type to use in host code
    using compatseed_container_buffer
    = container_buffer< int, float >;

    /// Convenience declaration for the compatseed container view type to use in host code
    using compatseed_container_view
    = container_view< int, float >;        
    
}// namespace cuda
}// namespace traccc
    
