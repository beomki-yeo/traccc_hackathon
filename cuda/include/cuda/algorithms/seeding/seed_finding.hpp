/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <algorithms/seeding/detail/seeding_config.hpp>
#include <iostream>

namespace traccc{
    
namespace cuda{

struct seed_finding{
    
seed_finding(seedfinder_config& config,
	     const host_internal_spacepoint_container& isp_container,
	     experiment_cuts* exp_cuts = nullptr,
	     vecmem::memory_resource* mr = nullptr):
    m_isp_container(isp_container),
    m_mr(mr)
{}
    
host_seed_collection operator()(){
    host_seed_collection seed_collection;
    this->operator()(seed_collection);
    
    return seed_collection;
}
        
void operator()(host_seed_collection& seeds){

    // allocate doublet container
    size_t n_bins = m_isp_container.headers.size();
    
    host_doublet_container mid_bot_container = {
	   vecmem::vector< size_t >(n_bins, 0, m_mr),
	   vecmem::jagged_vector< doublet >(n_bins, m_mr)
    };    
    
    host_doublet_container mid_top_container = {
           vecmem::vector< size_t >(n_bins, 0, m_mr),
	   vecmem::jagged_vector< doublet >(n_bins, m_mr)
    };

    
    
}

private:
    const host_internal_spacepoint_container& m_isp_container;
    vecmem::memory_resource* m_mr;
};        
    
} // namespace cuda
} // namespace traccc
