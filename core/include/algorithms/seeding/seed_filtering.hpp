/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <algorithms/seeding/detail/experiment_cuts.hpp>

#pragma once

namespace traccc{

struct seed_filtering{

seed_filtering(const host_internal_spacepoint_container& isp_container,
	       experiment_cuts* exp_cuts = nullptr)
    :m_isp_container(isp_container),
     m_exp_cuts(exp_cuts)
    {}
    
void operator()(host_triplet_collection& triplets,
		host_seed_collection& seeds){

    host_seed_collection seeds_per_spM;
    
    for (auto& triplet: triplets){
	if (m_exp_cuts != nullptr){
	    m_exp_cuts->seedWeight(triplet);
	    if (!m_exp_cuts->singleSeedCut(triplet)){
		continue;
	    }
	}
	
	// bottom
	auto spB_idx = triplet.sp1;
	auto spB = m_isp_container.items[spB_idx.bin_idx][spB_idx.sp_idx];
	
	// middle
	auto spM_idx = triplet.sp2; 
	auto spM = m_isp_container.items[spM_idx.bin_idx][spM_idx.sp_idx];
	
	// top
	auto spT_idx = triplet.sp3; 
	auto spT = m_isp_container.items[spT_idx.bin_idx][spT_idx.sp_idx];
	
	seeds_per_spM.push_back({spB,spM,spT,triplet.weight, triplet.z_vertex});
    }
    
    // sort seeds based on their weights	
    std::sort(seeds_per_spM.begin(), seeds_per_spM.end(),
	      [](seed& seed1, seed& seed2){
		  return (seed1.weight > seed2.weight);});
    if (m_exp_cuts != nullptr){
	seeds_per_spM = m_exp_cuts->cutPerMiddleSP(std::move(seeds_per_spM));
    }
    unsigned int maxSeeds = seeds_per_spM.size();
    
    if (maxSeeds > m_filter_config.maxSeedsPerSpM) {
	maxSeeds = m_filter_config.maxSeedsPerSpM + 1;
    }
    
    auto itBegin = seeds_per_spM.begin();
    auto it = seeds_per_spM.begin();
    // default filter removes the last seeds if maximum amount exceeded
    // ordering by weight by filterSeeds_2SpFixed means these are the lowest
    // weight seeds
    
    for (; it < itBegin + maxSeeds; ++it) {
	seeds.push_back(*it);
    }
    
}
    
private:    
    seedfilter_config m_filter_config;
    const host_internal_spacepoint_container& m_isp_container;
    std::shared_ptr<experiment_cuts> m_exp_cuts;
};

    
} // namespace traccc
