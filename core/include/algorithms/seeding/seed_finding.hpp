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
#include <algorithms/seeding/doublet_finding.hpp>
#include <algorithms/seeding/transform_coordinates.hpp>
#include <algorithms/seeding/triplet_finding.hpp>

namespace traccc{
    
struct seed_finding{

seed_finding(const seedfinder_config& config, const host_internal_spacepoint_container& isp_container):
    m_doublet_finding(config, isp_container),
    m_transform_coordinates(config, isp_container),
    m_triplet_finding(config, isp_container),
    m_isp_container(isp_container)
    {}
    
host_seed_collection operator()(){
    host_seed_collection seed_collection;
    this->operator()(seed_collection);
    
    return seed_collection;
}
        
void operator()(host_seed_collection& seed_collection){
    
    // iterate over grid bins
    for (size_t i=0; i<m_isp_container.headers.size(); ++i){
	auto bin_location = m_isp_container.headers[i];
	auto spM_collection = m_isp_container.items[i];
	
	/// iterate over middle spacepoints
	for (auto spM: spM_collection){
	    
	    /// 1. find doublets
	    auto compat_id = m_doublet_finding(bin_location, spM);
	    
	    auto& compat_bottom_id = compat_id.first;
	    auto& compat_top_id = compat_id.second;
	    
	    if (compat_bottom_id.empty() || compat_top_id.empty()){
		continue;
	    }
	    
	    /// 2. conformal transformation
	    auto lin_circles_bottom = m_transform_coordinates(bin_location, compat_bottom_id, spM, true);
	    auto lin_circles_top = m_transform_coordinates(bin_location, compat_top_id, spM, false);
	    
	    /// 3. find triplets
	    auto seeds = m_triplet_finding(lin_circles_bottom, lin_circles_top,
					   compat_bottom_id, compat_top_id, spM);
	    
	    
	}		
    }	    
}
    
private:
    seedfinder_config m_config;
    const host_internal_spacepoint_container& m_isp_container;
    doublet_finding m_doublet_finding;
    transform_coordinates m_transform_coordinates;
    triplet_finding m_triplet_finding;
};
} // namespace traccc
