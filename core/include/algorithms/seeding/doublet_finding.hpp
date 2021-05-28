/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

#include <edm/internal_spacepoint.hpp>
#include <algorithms/seeding/detail/doublet.hpp>

namespace traccc{

struct doublet_finding{
    
doublet_finding(vecmem::memory_resource& resource, seedfinder_config& config, const host_internal_spacepoint_container& isp_container):
    m_resource(&resource),
    m_config(config),
    m_isp_container(isp_container)
    {}
    
//host_doublet_collection operator()(const bin_information& bin_information,
std::vector<doublet> operator()(const bin_information& bin_information,
				const sp_location& spM_location,
				bool bottom){

    //host_doublet_collection doublets;
    //host_doublet_collection doublets(m_resource);
    std::vector<doublet> doublets;
    this->operator()(bin_information,
		     spM_location,
		     doublets,
		     bottom);
    
    return doublets;
}
    
void operator()(const bin_information& bin_information,
		const sp_location& spM_location,
		//host_doublet_collection& doublets,
		std::vector<doublet>& doublets,
		bool bottom){
    const auto& spM = m_isp_container.items[spM_location.bin_idx][spM_location.sp_idx];
    float rM = spM.radius();
    float zM = spM.z();
    float varianceRM = spM.varianceR();
    float varianceZM = spM.varianceZ();
    
	if (bottom){

	    auto& counts = bin_information.bottom_idx.counts;
	    auto& bottom_bin_indices = bin_information.bottom_idx.vector_indices;
	    
	    for (size_t i=0; i<counts; ++i){
		auto& bin_idx = bottom_bin_indices[i];
		auto& spacepoints = m_isp_container.items[bin_idx];
				
		for (size_t sp_idx=0; sp_idx < spacepoints.size(); ++sp_idx){
		    auto& spB = spacepoints[sp_idx];
		    		    
		    float rB = spB.radius();
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
		    float cotTheta = (zM - spB.z()) / deltaR;
		    if (std::fabs(cotTheta) > m_config.cotThetaMax) {
			continue;
		    }
		    // check if duplet origin on z axis within collision region
		    float zOrigin = zM - rM * cotTheta;
		    if (zOrigin < m_config.collisionRegionMin ||
			zOrigin > m_config.collisionRegionMax) {
			continue;
		    }	    
		    
		    sp_location spB_location = {bin_idx,sp_idx};
		    lin_circle lin = transform_coordinates(spM,spB,bottom);	    
		    doublets.push_back(doublet({spM_location,spB_location,lin}));
		}		
	    }	    
	} // if bottom == true		
	else if (!bottom){
	    
	    auto& counts = bin_information.top_idx.counts;
	    auto& top_bin_indices = bin_information.top_idx.vector_indices;

	    for (size_t i=0; i<counts; ++i){
		
		auto& bin_idx = top_bin_indices[i];
		auto& spacepoints = m_isp_container.items[bin_idx];
		
		for (size_t sp_idx=0; sp_idx < spacepoints.size(); ++sp_idx){
		    auto& spT = spacepoints[sp_idx];
  		    
		    float rT = spT.radius();
		    float deltaR = rT - rM;
		    // this condition is the opposite of the condition for bottom SP
		    if (deltaR < m_config.deltaRMin) {
			continue;
		    }
		    if (deltaR > m_config.deltaRMax) {
			continue;
		    }
		    
		    float cotTheta = (spT.z() - zM) / deltaR;
		    if (std::fabs(cotTheta) > m_config.cotThetaMax) {
			continue;
		    }
		    float zOrigin = zM - rM * cotTheta;
		    if (zOrigin < m_config.collisionRegionMin ||
			zOrigin > m_config.collisionRegionMax) {
			continue;
		    }
		    
		    sp_location spT_location = {bin_idx,sp_idx};
		    lin_circle lin = transform_coordinates(spM,spT,bottom);
		    doublets.push_back(doublet({spM_location, spT_location, lin}));
		}		
	    }
	    	
	} // if bottom == false	
    }

    inline
    lin_circle transform_coordinates(const internal_spacepoint<spacepoint>& sp1,
				     const internal_spacepoint<spacepoint>& sp2,
				     bool bottom){
	float xM = sp1.x();
	float yM = sp1.y();
	float zM = sp1.z();
	float rM = sp1.radius();
	float varianceZM = sp1.varianceZ();
	float varianceRM = sp1.varianceR();
	float cosPhiM = xM / rM;
	float sinPhiM = yM / rM;

	float deltaX = sp2.x() - xM;
	float deltaY = sp2.y() - yM;
	float deltaZ = sp2.z() - zM;
	// calculate projection fraction of spM->sp vector pointing in same
	// direction as
	// vector origin->spM (x) and projection fraction of spM->sp vector pointing
	// orthogonal to origin->spM (y)
	float x = deltaX * cosPhiM + deltaY * sinPhiM;
	float y = deltaY * cosPhiM - deltaX * sinPhiM;
	// 1/(length of M -> SP)
	float iDeltaR2 = 1. / (deltaX * deltaX + deltaY * deltaY);
	float iDeltaR = std::sqrt(iDeltaR2);
	//
	int bottomFactor = 1 * (int(!bottom)) - 1 * (int(bottom));
	// cot_theta = (deltaZ/deltaR)
	float cot_theta = deltaZ * iDeltaR * bottomFactor;
	// VERY frequent (SP^3) access
	lin_circle l;
	l.cotTheta = cot_theta;
	// location on z-axis of this SP-duplet
	l.Zo = zM - rM * cot_theta;
	l.iDeltaR = iDeltaR;
	// transformation of circle equation (x,y) into linear equation (u,v)
	// x^2 + y^2 - 2x_0*x - 2y_0*y = 0
	// is transformed into
	// 1 - 2x_0*u - 2y_0*v = 0
	// using the following m_U and m_V
	// (u = A + B*v); A and B are created later on
	l.U = x * iDeltaR2;
	l.V = y * iDeltaR2;
	// error term for sp-pair without correlation of middle space point
	l.Er = ((varianceZM + sp2.varianceZ()) +
		(cot_theta * cot_theta) * (varianceRM + sp2.varianceR())) *
	    iDeltaR2;

	return l;
    }
    
private:
    vecmem::memory_resource* m_resource;
    seedfinder_config m_config;
    const host_internal_spacepoint_container& m_isp_container;    
};

} // namespace traccc

