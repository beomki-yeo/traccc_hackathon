/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <algorithms/seeding/detail/triplet.hpp>
#include <cuda/algorithms/seeding/detail/triplet_counter.hpp>
#include <algorithms/seeding/detail/seeding_config.hpp>

namespace traccc{    
namespace cuda{

void weight_updating(const seedfilter_config& filter_config,
		     host_internal_spacepoint_container& internal_sp_container,
		     host_triplet_counter_container& triplet_counter_container,
		     host_triplet_container& triplet_container,
		     vecmem::memory_resource* resource
		     );

}// namespace cuda
}// namespace traccc