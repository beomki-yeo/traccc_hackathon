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
#include <cuda/algorithms/seeding/detail/stats_config.hpp>
#include <cuda/algorithms/seeding/detail/doublet_counter.hpp>
#include <cuda/algorithms/seeding/doublet_counting.cuh>
#include <cuda/algorithms/seeding/triplet_counting.cuh>
#include <cuda/algorithms/seeding/doublet_finding.cuh>
#include <cuda/algorithms/seeding/triplet_finding.cuh>
#include <cuda/algorithms/seeding/weight_updating.cuh>
#include <cuda/algorithms/seeding/seed_selecting.cuh>

#include <iostream>
#include <algorithm>

namespace traccc{
    
namespace cuda{

struct seed_finding{
    
seed_finding(seedfinder_config& config,
	     std::shared_ptr<spacepoint_grid> sp_grid,
	     stats_config* stats_cfg,
	     experiment_cuts* exp_cuts = nullptr,
	     vecmem::memory_resource* mr = nullptr):    
    m_seedfinder_config(config),
    m_seed_filtering(exp_cuts),
    m_sp_grid(sp_grid),
    m_stats_config(stats_cfg),
    m_mr(mr),

    doublet_counter_container({host_doublet_counter_container::header_vector(sp_grid->size(false),0, mr),
			       host_doublet_counter_container::item_vector(sp_grid->size(false),mr)}),
    
    mid_bot_container({host_doublet_container::header_vector(sp_grid->size(false),0, mr),
		       host_doublet_container::item_vector(sp_grid->size(false),mr)}),
    
    mid_top_container({host_doublet_container::header_vector(sp_grid->size(false), 0, mr),
		       host_doublet_container::item_vector(sp_grid->size(false),mr)}),

    triplet_counter_container({host_triplet_counter_container::header_vector(sp_grid->size(false),0, mr),
			       host_triplet_counter_container::item_vector(sp_grid->size(false),mr)}),
    
    triplet_container({host_triplet_container::header_vector(sp_grid->size(false), 0, mr),
		       host_triplet_container::item_vector(sp_grid->size(false),mr)}),
    seed_container({host_seed_container::header_vector(1, 0, mr),
		    host_seed_container::item_vector(1,mr)})
    {
	first_alloc = true;
    }
    
host_seed_collection operator()(host_internal_spacepoint_container& isp_container){
    
    // initialize multiplet container

    size_t n_internal_sp = 0;
    for (size_t i=0; i<isp_container.headers.size(); ++i){
	size_t n_spM = isp_container.items[i].size();
	size_t n_mid_bot_doublets = m_stats_config->get_mid_bot_doublets_size(n_spM);
	size_t n_mid_top_doublets = m_stats_config->get_mid_top_doublets_size(n_spM);
	size_t n_triplets = m_stats_config->get_triplets_size(n_spM);

	///// Zero initialization
	doublet_counter_container.headers[i] = 0;
	mid_bot_container.headers[i] = 0;
	mid_top_container.headers[i] = 0;
	triplet_counter_container.headers[i] = 0;
	triplet_container.headers[i] = 0;
	
	doublet_counter_container.items[i].resize(n_spM);	
	mid_bot_container.items[i].resize(n_mid_bot_doublets);       
	mid_top_container.items[i].resize(n_mid_top_doublets);	    
	triplet_counter_container.items[i].resize(n_mid_bot_doublets);
	triplet_container.items[i].resize(n_triplets);

	n_internal_sp += isp_container.items[i].size();
    }	

    seed_container.headers[0] = 0;
    seed_container.items[0].resize(m_stats_config->get_seeds_size(n_internal_sp));
    
    first_alloc = false;
    
    host_seed_collection seed_collection;
    this->operator()(isp_container, seed_collection);
    
    return seed_collection;
}
        
void operator()(host_internal_spacepoint_container& isp_container,
		host_seed_collection& seeds){

    traccc::cuda::doublet_counting(m_seedfinder_config,
				   isp_container,
				   doublet_counter_container,
				   m_mr);
    
    traccc::cuda::doublet_finding(m_seedfinder_config,
				  isp_container,
				  doublet_counter_container,
				  mid_bot_container,
				  mid_top_container,
				  m_mr);
   
    traccc::cuda::triplet_counting(m_seedfinder_config,
				   m_seedfilter_config,
				   isp_container,
				   doublet_counter_container,
				   mid_bot_container,
				   mid_top_container,
				   triplet_counter_container,
				   m_mr);
    
    traccc::cuda::triplet_finding(m_seedfinder_config,
				  m_seedfilter_config,
				  isp_container,
				  doublet_counter_container,
				  mid_bot_container,
				  mid_top_container,
				  triplet_counter_container,
				  triplet_container,
				  m_mr);
    
    traccc::cuda::weight_updating(m_seedfilter_config,
				  isp_container,
				  triplet_counter_container,
				  triplet_container,
				  m_mr);

    /*
    traccc::cuda::seed_filtering(m_seedfilter_config,
				 isp_container,
				 doublet_counter,
				 triplet_counter,
				 triplet_container,
				 seed_container,
				 m_mr);
    */

    
    for(size_t i=0; i < m_sp_grid->size(false); ++i){
	// Get triplets per spM
	
	auto n_triplets = triplet_container.headers[i];
	
	auto triplets = triplet_container.items[i];
		
	triplets.erase(triplets.begin()+n_triplets,
		       triplets.end());
	
	std::sort(triplets.begin(), triplets.end(),
		  [](triplet& t1, triplet& t2){
		      if (t1.sp2.sp_idx < t2.sp2.sp_idx) return true;
		      if (t2.sp2.sp_idx < t1.sp2.sp_idx) return false;
		  });		
	
	auto last = std::unique(triplets.begin(), triplets.end(),
				[] (triplet const & lhs, triplet const & rhs) {
				    return (lhs.sp2.sp_idx == rhs.sp2.sp_idx);
				}
				);		
	
	for (auto it = triplets.begin(); it != last; it++){
	    host_triplet_collection triplet_per_spM;
	    
	    for (int j=0; j<n_triplets; j++){
		auto& triplet = triplet_container.items[i][j];
		
		if (triplet.sp2.sp_idx == it->sp2.sp_idx){
		    triplet_per_spM.push_back(triplet);
		}
	    }

	    if (triplet_per_spM.size() > 0){
		m_seed_filtering(isp_container, triplet_per_spM, seeds);
	    }
	}	
    }           
}

private:

    bool first_alloc;
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
    std::shared_ptr< spacepoint_grid > m_sp_grid;    
    stats_config* m_stats_config;
    seed_filtering m_seed_filtering;

    host_doublet_counter_container doublet_counter_container;
    host_doublet_container mid_bot_container;
    host_doublet_container mid_top_container;
    host_triplet_counter_container triplet_counter_container;
    host_triplet_container triplet_container;
    host_seed_container seed_container;
    vecmem::memory_resource* m_mr;
};        

} // namespace cuda
} // namespace traccc
