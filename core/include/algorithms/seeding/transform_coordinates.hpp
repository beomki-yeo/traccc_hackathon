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

struct transform_coordinates{

transform_coordinates(const seedfinder_config& config, const host_internal_spacepoint_container& isp_container):
    m_config(config),
    m_isp_container(isp_container)
    {}
    
std::vector<lin_circle> operator()(const bin_information& bin_information,
				   const std::vector< sp_location >& compat_id,
				   const internal_spacepoint<spacepoint> spM,
				   bool bottom){
    
    std::vector < lin_circle > lin_circles;
    
    this->operator()(bin_information,
		     compat_id,
		     spM,
		     bottom,
		     lin_circles);
    
    return lin_circles;	
    
}
    
void operator()(const bin_information& bin_information,
		const std::vector< sp_location >& compat_id,
		const internal_spacepoint<spacepoint> spM,
		bool bottom,
		std::vector< lin_circle >& lin_circles){	    	    
    
    float xM = spM.x();
    float yM = spM.y();
    float zM = spM.z();
    float rM = spM.radius();
    float varianceZM = spM.varianceZ();
    float varianceRM = spM.varianceR();
    float cosPhiM = xM / rM;
    float sinPhiM = yM / rM;	
    
    if (bottom==true){
	auto& bin_indices = bin_information.bottom_idx.vector_indices;
	
	for (auto bin_idx: bin_indices){
	    for (size_t sp_idx=0; sp_idx<m_isp_container.items[bin_idx].size(); ++sp_idx){
		auto& sp = m_isp_container.items[bin_idx][sp_idx];
		float deltaX = sp.x() - xM;
		float deltaY = sp.y() - yM;
		float deltaZ = sp.z() - zM;
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
		l.Er = ((varianceZM + sp.varianceZ()) +
			(cot_theta * cot_theta) * (varianceRM + sp.varianceR())) *
		    iDeltaR2;
		lin_circles.push_back(l);		
	    }
	}
    }    
}

private:
    seedfinder_config m_config;
    const host_internal_spacepoint_container& m_isp_container;    
};

} // namespace traccc
