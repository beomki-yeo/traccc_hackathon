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
#include <algorithm>

namespace traccc{
    
namespace cuda{

struct seed_finding{
    
seed_finding(seedfinder_config& config,
	     std::shared_ptr<spacepoint_grid> sp_grid,
	     multiplet_config* multi_cfg,
	     experiment_cuts* exp_cuts = nullptr,
	     vecmem::memory_resource* mr = nullptr):    
    m_seedfinder_config(config),
    m_sp_grid(sp_grid),
    m_multiplet_config(multi_cfg),
    m_mr(mr),
    
    mid_bot_container({host_doublet_container::header_vector(sp_grid->size(false),0, mr),
		       host_doublet_container::item_vector(sp_grid->size(false),mr)}),
    
    mid_top_container({host_doublet_container::header_vector(sp_grid->size(false), 0, mr),
		       host_doublet_container::item_vector(sp_grid->size(false),mr)}),
    
    triplet_container({host_triplet_container::header_vector(sp_grid->size(false), 0, mr),
		       host_triplet_container::item_vector(sp_grid->size(false),mr)})
       
    {}
    
host_seed_collection operator()(host_internal_spacepoint_container& isp_container){
    
    // initialize multiplet container
    for (size_t i=0; i<isp_container.headers.size(); ++i){
	size_t n_spM = isp_container.items[i].size();
	size_t n_mid_bot_doublets = m_multiplet_config->get_mid_bot_doublets_size(n_spM);
	size_t n_mid_top_doublets = m_multiplet_config->get_mid_top_doublets_size(n_spM);
	size_t n_triplets = m_multiplet_config->get_triplets_size(n_spM);

	mid_bot_container.headers[i] = 0;
	mid_top_container.headers[i] = 0;
	triplet_container.headers[i] = 0;
	
	mid_bot_container.items[i].resize(n_mid_bot_doublets);       
	mid_top_container.items[i].resize(n_mid_top_doublets);	
	triplet_container.items[i].resize(n_triplets);
    }	
    
    host_seed_collection seed_collection;
    this->operator()(isp_container, seed_collection);
    
    return seed_collection;
}
        
void operator()(host_internal_spacepoint_container& isp_container,
		host_seed_collection& seeds){
    
    traccc::cuda::doublet_finding(m_seedfinder_config,
				  isp_container,
				  mid_bot_container,
				  mid_top_container,
				  m_mr);

    // sort doublets in terms of middle spacepoint idx
    // note: it takes too long with GPU bubble sort so used cpu sort function
    for (int i=0; i<mid_bot_container.headers.size(); ++i){
	auto n_doublets = mid_bot_container.headers[i];
	auto& doublets = mid_bot_container.items[i];
	std::sort(doublets.begin(), doublets.begin()+n_doublets,
		  [](doublet& d1, doublet& d2){
		      return d1.sp1.sp_idx < d2.sp1.sp_idx;
		  });		
    }
        
    for (int i=0; i<mid_top_container.headers.size(); ++i){
	auto n_doublets = mid_top_container.headers[i];
	auto& doublets = mid_top_container.items[i];
	std::sort(doublets.begin(), doublets.begin()+n_doublets,
		  [](doublet& d1, doublet& d2){
		      return d1.sp1.sp_idx < d2.sp1.sp_idx;
		  });		
    }

    // get the size of doublets with same spM
    vecmem::jagged_vector< std::pair<size_t, size_t > > n_doublets_per_bin(m_sp_grid->size(false),m_mr);

    for(size_t i=0; i < m_sp_grid->size(false); ++i){
	auto n_mid_bot_doublets = mid_bot_container.headers[i];
	auto mid_bot_doublets = mid_bot_container.items[i];
	mid_bot_doublets.erase(mid_bot_doublets.begin()+n_mid_bot_doublets,
			       mid_bot_doublets.end());

	auto n_mid_top_doublets = mid_top_container.headers[i];
	auto mid_top_doublets = mid_top_container.items[i];
	mid_top_doublets.erase(mid_top_doublets.begin()+n_mid_top_doublets,
			       mid_top_doublets.end());
	
	auto mb_last = std::unique(mid_bot_doublets.begin(), mid_bot_doublets.end(),
				   [] (doublet const & lhs, doublet const & rhs) {
				    return (lhs.sp1.sp_idx == rhs.sp1.sp_idx);
				   }
				   );
	/*
	std::cout << "mid bot: " << std::endl;
	for (size_t j=0; j<n_mid_bot_doublets; ++j){
	    std::cout << mid_bot_container.items[i][j].sp1.sp_idx << " "; 
	}
	std::cout << std::endl;
	std::cout << "mid top: " << std::endl;
	for (size_t j=0; j<n_mid_top_doublets; ++j){
	    std::cout << mid_top_container.items[i][j].sp1.sp_idx << " "; 
	}	
	std::cout << std::endl;
	*/

	n_doublets_per_bin[i].reserve(mb_last-mid_bot_doublets.begin());
	
	for (auto it = mid_bot_doublets.begin(); it != mb_last; it++){
	    auto mid_bot_size = std::count_if(
		  mid_bot_container.items[i].begin(),
		  mid_bot_container.items[i].begin()+n_mid_bot_doublets,
		  [&](const doublet& d) {
		      return d.sp1.sp_idx == it->sp1.sp_idx;
		  });
	    
	    auto mid_top_size = std::count_if(
		  mid_top_container.items[i].begin(),
		  mid_top_container.items[i].begin()+n_mid_top_doublets,
		  [&](const doublet& d) {
		      return d.sp1.sp_idx == it->sp1.sp_idx;
		  });

	    n_doublets_per_bin[i].push_back(std::make_pair(mid_bot_size,mid_top_size));	    
	    //std::cout << it->sp1.sp_idx << "  " << mid_bot_size << "  " << mid_top_size << std::endl;
	}			
    }

    /*
    traccc::cuda::triplet_finding(m_seedfinder_config,
				  m_seedfilter_config,
				  isp_container,
				  mid_bot_container,
				  mid_top_container,
				  n_doublets_per_bin,
				  m_mr);
    */
}

private:
    const seedfinder_config m_seedfinder_config;
    const seedfinder_config m_seedfilter_config;
    std::shared_ptr< spacepoint_grid > m_sp_grid;    
    multiplet_config* m_multiplet_config;
    
    host_doublet_container mid_bot_container;
    host_doublet_container mid_top_container;
    host_triplet_container triplet_container;
        
    vecmem::memory_resource* m_mr;
};        
    
} // namespace cuda
} // namespace traccc
