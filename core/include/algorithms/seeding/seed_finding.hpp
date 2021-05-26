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
#include <algorithms/seeding/triplet_finding.hpp>
#include <algorithms/seeding/seed_filtering.hpp>
#include <iostream>

namespace traccc{
    
struct seed_finding{

seed_finding(const seedfinder_config& config,
	     const host_internal_spacepoint_container& isp_container,
	     experiment_cuts* exp_cuts = nullptr):
    m_doublet_finding(config, isp_container),
    m_triplet_finding(config, isp_container),
    m_seed_filtering(isp_container, exp_cuts),
    m_isp_container(isp_container)
    {
	// calculation of scattering using the highland formula
	// convert pT to p once theta angle is known
	m_config.highland = 13.6 * std::sqrt(m_config.radLengthPerSeed) *
	    (1 + 0.038 * std::log(m_config.radLengthPerSeed));
	float maxScatteringAngle = m_config.highland / m_config.minPt;
	m_config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
	// helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
	// millimeter
	// TODO: change using ACTS units
	m_config.pTPerHelixRadius = 300. * m_config.bFieldInZ;
	m_config.minHelixDiameter2 =
	    std::pow(m_config.minPt * 2 / m_config.pTPerHelixRadius, 2);
	m_config.pT2perRadius =
	    std::pow(m_config.highland / m_config.pTPerHelixRadius, 2);		
    }
    
host_seed_collection operator()(){
    host_seed_collection seed_collection;
    this->operator()(seed_collection);
    
    return seed_collection;
}
        
void operator()(host_seed_collection& seeds){

    // iterate over grid bins
    for (size_t i=0; i<m_isp_container.headers.size(); ++i){
	
	auto bin_location = m_isp_container.headers[i];
	auto spM_collection = m_isp_container.items[i];

	/// iterate over middle spacepoints
	for (size_t j=0; j<spM_collection.size(); ++j){
	    
	    sp_location spM_location({i,j});
	   
	    // doublet search
	    auto doublets_mid_bot = m_doublet_finding(bin_location, spM_location, true);
	    if (doublets_mid_bot.empty()) continue;
	    
	    auto doublets_mid_top = m_doublet_finding(bin_location, spM_location, false);
	    if (doublets_mid_top.empty()) continue;
	    
	    host_triplet_collection triplets_per_spM;
	    
	    for (auto mid_bot: doublets_mid_bot){
		m_triplet_finding(mid_bot, doublets_mid_top, triplets_per_spM);	 
	    }
	    
	    m_seed_filtering(triplets_per_spM, seeds);
	}

    }
}
    
private:
    seedfinder_config m_config;
    const host_internal_spacepoint_container& m_isp_container;
    doublet_finding m_doublet_finding;
    triplet_finding m_triplet_finding;
    seed_filtering m_seed_filtering;
};
} // namespace traccc
