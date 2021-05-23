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
#include <iostream>

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

	size_t n_doublets_mid_bot = 0;
	size_t n_doublets_mid_top = 0;
	
	/// iterate over middle spacepoints
	for (size_t j=0; j<spM_collection.size(); ++j){
	    sp_location spM_location({i,j});

	    auto doublets_mid_bot = m_doublet_finding(bin_location, spM_location, true);
	    auto doublets_mid_top = m_doublet_finding(bin_location, spM_location, false);

	    n_doublets_mid_bot += doublets_mid_bot.size();
	    n_doublets_mid_top += doublets_mid_top.size();

	    //std::cout << doublets_mid_bot.size() << std::endl;
	}

	std::cout << i << ",   " << spM_collection.size() << ",   " << n_doublets_mid_bot << ",   " << n_doublets_mid_top << std::endl;
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
