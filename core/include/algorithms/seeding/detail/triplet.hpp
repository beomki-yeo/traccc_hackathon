/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

#include <algorithms/seeding/detail/doublet.hpp>

namespace traccc{

    // Bottom - Middle - Top
    struct triplet{
	sp_location sp1; // bottom
	sp_location sp2; // middle
	sp_location sp3; // top
	scalar curvature;
	scalar impact_parameter;
	scalar weight;
	scalar z_vertex;
    };    
    
    /// Container of triplet belonging to one detector module
    template< template< typename > class vector_t >
    using triplet_collection = vector_t< triplet >;

    /// Convenience declaration for the triplet collection type to use in host code
    using host_triplet_collection
    = triplet_collection< vecmem::vector >;

    /// Convenience declaration for the triplet collection type to use in device code
    using device_triplet_collection
    = triplet_collection< vecmem::device_vector >;

    /// Convenience declaration for the triplet container type to use in host code
    using host_triplet_container
    = host_container< size_t, triplet >;

    /// Convenience declaration for the triplet container type to use in device code
    using device_triplet_container
    = device_container< size_t, triplet>;

    /// Convenience declaration for the triplet container data type to use in host code
    using triplet_container_data
    = container_data< size_t, triplet >;

    /// Convenience declaration for the triplet container buffer type to use in host code
    using triplet_container_buffer
    = container_buffer< size_t, triplet >;

    /// Convenience declaration for the triplet container view type to use in host code
    using triplet_container_view
    = container_view< size_t, triplet >;        
    
} // namespace traccc
