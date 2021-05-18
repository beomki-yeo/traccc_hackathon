/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/container.hpp"

namespace traccc{

    template < typename spacepoint >
    struct seed{
	float z_vertex;
	std::array< spacepoint, 3u > spacepoints;
    };

    template< template< typename > class vector_t >
    using seed_collection = vector_t< seed < spacepoint > >;

    using host_seed_collection = seed_collection< vecmem::vector >;

    using device_seed_collection = seed_collection< vecmem::device_vector >;
    
}; //namespace traccc


