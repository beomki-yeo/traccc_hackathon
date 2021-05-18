/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

#include <edm/internal_spacepoint.hpp>
#include <algorithms/seeding/detail/multiplet.hpp>

namespace traccc{

    struct doublet_finding{

	doublet_finding(const seedfinder_config& config): m_config(config){}

	host_doublet_collection operator()(const host_internal_spacepoint_collection& isp_collection){
	    host_doublet_collection doublet_collection;
	    this->operator()(isp_collection, doublet_collection);
	    
	    return doublet_collection;
	}

	void operator()(const host_internal_spacepoint_collection& isp_collection,
			host_doublet_collection& doublet_collection){
	    for (auto spM: isp_collection){
		float rM = spM.radius();
		float zM = spM.z();
		float varianceRM = spM.varianceR();
		float varianceZM = spM.varianceZ();

		
		
	    }	    
	}
	
    private:
	seedfinder_config m_config;
	
    };

} // namespace traccc

