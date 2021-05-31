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
#include <cuda/algorithms/seeding/detail/multiplet_config.hpp>
#include <cuda/algorithms/seeding/doublet_finding.cuh>

#include <iostream>

namespace traccc{
    
namespace cuda{

struct seed_finding{
    
seed_finding(seedfinder_config& config,
	     host_internal_spacepoint_container& isp_container,
	     multiplet_config* multi_cfg,
	     experiment_cuts* exp_cuts = nullptr,
	     vecmem::memory_resource* mr = nullptr):    
    m_isp_container(isp_container),
    m_seedfinder_config(config),
    m_multiplet_config(multi_cfg),
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

    host_triplet_container triplet_container = {
           vecmem::vector< size_t >(n_bins, 0, m_mr),
	   vecmem::jagged_vector< triplet >(n_bins, m_mr)
    };
    
    for (size_t i=0; i<n_bins; ++i){
	size_t n_spM = m_isp_container.items[i].size();
	size_t n_mid_bot_doublets = m_multiplet_config->get_mid_bot_doublets_size(n_spM);
	size_t n_mid_top_doublets = m_multiplet_config->get_mid_top_doublets_size(n_spM);
	size_t n_triplets = m_multiplet_config->get_triplets_size(n_spM);
	
	mid_bot_container.items[i] = vecmem::vector<doublet>(n_mid_bot_doublets);
	mid_top_container.items[i] = vecmem::vector<doublet>(n_mid_top_doublets);
	triplet_container.items[i] = vecmem::vector<triplet>(n_triplets);    
    }

    traccc::cuda::doublet_finding(m_seedfinder_config, m_isp_container, mid_bot_container, true, m_mr);
    
    
}

private:
    host_internal_spacepoint_container& m_isp_container;
    multiplet_config* m_multiplet_config;
    const seedfinder_config m_seedfinder_config;
    vecmem::memory_resource* m_mr;
};        
    
} // namespace cuda
} // namespace traccc
