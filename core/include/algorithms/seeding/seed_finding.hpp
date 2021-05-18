/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <algorithms/seeding/detail/multiplet.hpp>
#include <algorithms/seeding/detail/seeding_config.hpp>
#include <algorithms/seeding/doublet_finding.hpp>

namespace traccc{

    struct seed_finding{

	seed_finding(const seedfinder_config& config)
	    : m_df(config)
	{}
	
	host_seed_collection operator()(const host_internal_spacepoint_container& isp_container){
	    host_seed_collection seed_collection;
	    this->operator()(isp_container);

	    return seed_collection;
	}


	void operator()(const host_internal_spacepoint_container& isp_container,
			host_seed_collection& seed_collection){

	    // iterate over bins
	    for (size_t i=0; i<isp_container.headers.size(); ++i){
		auto bin_location = isp_container.headers[i];
		auto spM_collection = isp_container.items[i];
				
		// find doublets
		//auto doublet_collection = m_df(isp_container, bin_location, spM_collection);
	    }	    
	}
	
    private:
	seedfinder_config m_config;
	doublet_finding m_df;
    };
} // namespace traccc
