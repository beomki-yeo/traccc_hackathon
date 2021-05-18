/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/container.hpp"
#include <vector>

namespace traccc {
    
    struct sp_location{
	/// index of the bin of the spacepoint grid
	/// (for header of internal spacepoint EDM)
	size_t bin_id;
	/// index of the spacepoint in the bin
	/// (for item of internal spacepoint EDM)
	size_t sp_id; 
    };

    using doublet = std::array<sp_location, 2>;
    using triplet = std::array<sp_location, 3>;
    
    template< template< typename > class vector_t >
    using doublet_collection = vector_t< doublet >;

    using host_doublet_collection = doublet_collection< vecmem::vector >;

    using device_doublet_collection = doublet_collection< vecmem::device_vector >;

    using host_doublet_container = host_container< sp_location, doublet >;

    using device_doublet_container = device_container< sp_location, doublet >;

    using doublet_container_data = container_data< sp_location, doublet >;

    using doublet_container_buffer = container_buffer< sp_location, doublet >;
    
    using doublet_container_view = container_view< sp_location, doublet >;    

    /// triplet

    template< template< typename > class vector_t >
    using triplet_collection = vector_t< triplet >;

    using host_triplet_collection = triplet_collection< vecmem::vector >;

    using device_triplet_collection = triplet_collection< vecmem::device_vector >;

    using host_triplet_container = host_container< sp_location, triplet >;

    using device_triplet_container = device_container< sp_location, triplet >;

    using triplet_container_data = container_data< sp_location, triplet >;

    using triplet_container_buffer = container_buffer< sp_location, triplet >;
    
    using triplet_container_view = container_view< sp_location, triplet >;    
    
} // namespace traccc
