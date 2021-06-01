/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

#include <edm/internal_spacepoint.hpp>
#include <algorithms/seeding/detail/doublet.hpp>
#include <algorithms/seeding/doublet_finding_helper.hpp>

namespace traccc{

struct doublet_finding{
    
doublet_finding(seedfinder_config& config, const host_internal_spacepoint_container& isp_container):
    m_config(config),
    m_isp_container(isp_container)
    {}
    
host_doublet_collection operator()(const bin_information& bin_information,
				   const sp_location& spM_location,
				   bool bottom){

    host_doublet_collection doublets;

    this->operator()(bin_information,
		     spM_location,
		     doublets,
		     bottom);
    
    return doublets;
}
    
void operator()(const bin_information& bin_information,
		const sp_location& spM_location,
		host_doublet_collection& doublets,
		bool bottom){
    const auto& spM = m_isp_container.items[spM_location.bin_idx][spM_location.sp_idx];
    float rM = spM.radius();
    float zM = spM.z();
    float varianceRM = spM.varianceR();
    float varianceZM = spM.varianceZ();
	
    auto& counts = bin_information.bottom_idx.counts;
    auto& bottom_bin_indices = bin_information.bottom_idx.vector_indices;

    if (bottom){
    
	for (size_t i=0; i<counts; ++i){
	    auto& bin_idx = bottom_bin_indices[i];
	    auto& spacepoints = m_isp_container.items[bin_idx];
	    
	    for (size_t sp_idx=0; sp_idx < spacepoints.size(); ++sp_idx){
		
		auto& spB = spacepoints[sp_idx];

		if (!doublet_finding_helper::isCompatible(spM,spB,m_config,bottom)){
		    continue;
		}
		
		lin_circle lin = doublet_finding_helper::transform_coordinates(spM,spB,bottom);
		sp_location spB_location = {bin_idx,sp_idx};
		doublets.push_back(doublet({spM_location,spB_location,lin}));
	    }		
	}
    }
    
    else if (!bottom){
	
	auto& counts = bin_information.top_idx.counts;
	auto& top_bin_indices = bin_information.top_idx.vector_indices;
	
	for (size_t i=0; i<counts; ++i){
	    
	    auto& bin_idx = top_bin_indices[i];
	    auto& spacepoints = m_isp_container.items[bin_idx];
	    
	    for (size_t sp_idx=0; sp_idx < spacepoints.size(); ++sp_idx){
		auto& spT = spacepoints[sp_idx];

		if (!doublet_finding_helper::isCompatible(spM,spT,m_config,bottom)){
		    continue;
		}

		lin_circle lin = doublet_finding_helper::transform_coordinates(spM,spT,bottom);
		sp_location spT_location = {bin_idx,sp_idx};		
		doublets.push_back(doublet({spM_location, spT_location, lin}));
	    }		
	}	
    }
}
    
private:
    vecmem::memory_resource* m_resource;
    seedfinder_config m_config;
    const host_internal_spacepoint_container& m_isp_container;    
};

} // namespace traccc

