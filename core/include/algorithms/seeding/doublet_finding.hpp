/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

#include <edm/internal_spacepoint.hpp>
#include <algorithms/seeding/detail/definitions.hpp>

namespace traccc{

struct doublet_finding{
    
    doublet_finding(const seedfinder_config& config, const host_internal_spacepoint_container& isp_container):
	m_config(config),
	m_isp_container(isp_container)
    {}
    
    std::pair< std::vector< sp_location >, std::vector< sp_location > > operator()(
	const bin_information& bin_information,
	const internal_spacepoint<spacepoint>& spM){
	
	std::vector < sp_location > compat_bottom_id;
	std::vector < sp_location > compat_top_id;
	
	this->operator()(bin_information,
			 spM,
			 compat_bottom_id,
			 compat_top_id);
	
	return std::make_pair(compat_bottom_id, compat_top_id);
    }
    
    void operator()(const bin_information& bin_information,
		    const internal_spacepoint<spacepoint>& spM,
		    std::vector< sp_location >& compat_bottom_id,
		    std::vector< sp_location >& compat_top_id){	    	    
	float rM = spM.radius();
	float zM = spM.z();
	float varianceRM = spM.varianceR();
	float varianceZM = spM.varianceZ();
	
	auto& bottom_bin_indices = bin_information.bottom_idx.vector_indices;
	
	for (auto bin_idx: bottom_bin_indices){
	    for (size_t sp_idx=0; sp_idx<m_isp_container.items[bin_idx].size(); ++sp_idx){
		auto& bottom_sp = m_isp_container.items[bin_idx][sp_idx];
		
		float rB = bottom_sp.radius();
		float deltaR = rM - rB;
		// if r-distance is too big, try next SP in bin
		if (deltaR > m_config.deltaRMax) {
		    continue;
		}
		// if r-distance is too small, continue because bins are NOT r-sorted
		if (deltaR < m_config.deltaRMin) {
		    continue;
		}
		// ratio Z/R (forward angle) of space point duplet
		float cotTheta = (zM - bottom_sp.z()) / deltaR;
		if (std::fabs(cotTheta) > m_config.cotThetaMax) {
		    continue;
		}
		// check if duplet origin on z axis within collision region
		float zOrigin = zM - rM * cotTheta;
		if (zOrigin < m_config.collisionRegionMin ||
		    zOrigin > m_config.collisionRegionMax) {
		    continue;
		}	    
		compat_bottom_id.push_back({bin_idx,sp_idx});	    
	    }
	}

	// terminate if there is no compatible bottom sp
	if (compat_bottom_id.empty()) return;
	
	auto& top_bin_indices = bin_information.top_idx.vector_indices;
	
	for (auto bin_idx: top_bin_indices){
	    for (size_t sp_idx=0; sp_idx<m_isp_container.items[bin_idx].size(); ++sp_idx){
		auto& top_sp = m_isp_container.items[bin_idx][sp_idx];	    

		float rT = top_sp.radius();
		float deltaR = rT - rM;
		// this condition is the opposite of the condition for bottom SP
		if (deltaR < m_config.deltaRMin) {
		    continue;
		}
		if (deltaR > m_config.deltaRMax) {
		    continue;
		}
		
		float cotTheta = (top_sp.z() - zM) / deltaR;
		if (std::fabs(cotTheta) > m_config.cotThetaMax) {
		    continue;
		}
		float zOrigin = zM - rM * cotTheta;
		if (zOrigin < m_config.collisionRegionMin ||
		    zOrigin > m_config.collisionRegionMax) {
		    continue;
		}
		compat_top_id.push_back({bin_idx,sp_idx});	
	    }
	}	    
    }
    
private:
    seedfinder_config m_config;
    const host_internal_spacepoint_container& m_isp_container;    
};

} // namespace traccc

