/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/container.hpp"

namespace traccc{

    struct seed{
	internal_spacepoint<spacepoint> spB;
	internal_spacepoint<spacepoint> spM;
	internal_spacepoint<spacepoint> spT;
	float weight;
	float z_vertex;

	seed& operator=(const seed& aSeed){
	    spB = aSeed.spB;
	    spM = aSeed.spM;
	    spT = aSeed.spT;	    
	    weight = aSeed.weight;
	    z_vertex = aSeed.z_vertex;
	    return *this;
	}
    };

    template< template< typename > class vector_t >
    using seed_collection = vector_t< seed >;

    using host_seed_collection = seed_collection< vecmem::vector >;

    using device_seed_collection = seed_collection< vecmem::device_vector >;

    using host_seed_container
    = host_container< bin_information, seed >;

    using device_seed_container
    = device_container< bin_information, seed >;

    using seed_container_data
    = container_data< bin_information, seed >;

    using seed_container_buffer
    = container_buffer< bin_information, seed >;

    using seed_container_view
    = container_view< bin_information, seed >;
    
}; //namespace traccc


