/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <algorithms/seeding/detail/doublet.hpp>
#include <algorithms/seeding/detail/seeding_config.hpp>
#include <algorithms/seeding/triplet_finding_helper.hpp>

namespace traccc{    
namespace cuda{

void triplet_finding(const seedfinder_config& config,
		     const seedfilter_config& filter_config,
		     host_internal_spacepoint_container& internal_sp_container,
		     host_doublet_container& mid_bot_doublet_container,
		     host_doublet_container& mid_top_doublet_container,
		     //vecmem::jagged_vector< std::pair<size_t, size_t> > n_doublets_per_spM,
		     vecmem::jagged_vector< size_t >& n_doublets_per_spM,
		     host_triplet_container& triplet_container,
		     vecmem::memory_resource* resource);
       
}// namespace cuda
}// namespace traccc
