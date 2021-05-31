/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */
#include "algorithms/seeding/detail/experiment_cuts.hpp"

#include <algorithm>

namespace traccc {
class atlas_cuts: public experiment_cuts {
public:
    
    atlas_cuts(const host_internal_spacepoint_container& isp_container):
	m_isp_container(isp_container)
    {}
    
    void seedWeight(triplet& aTriplet) const;
    
    bool singleSeedCut(const triplet& aTriplet) const;

    host_seed_collection cutPerMiddleSP(const host_seed_collection& seeds) const;
    
private:
    const host_internal_spacepoint_container& m_isp_container;    
};

void atlas_cuts::seedWeight(triplet& aTriplet) const {
    float weight = 0;
    
    // bottom
    auto& spB_idx = aTriplet.sp1;
    auto& spB = m_isp_container.items[spB_idx.bin_idx][spB_idx.sp_idx];   
    
    // top
    auto& spT_idx = aTriplet.sp3; 
    auto& spT = m_isp_container.items[spT_idx.bin_idx][spT_idx.sp_idx];

    if (spB.radius() > 150) {
	weight = 400;
    }
    if (spT.radius() < 150) {
	weight = 200;
    }

    aTriplet.weight += weight;
    return;
}

bool atlas_cuts::singleSeedCut(const triplet& aTriplet) const {
    // bottom
    auto& spB_idx = aTriplet.sp1;
    auto& spB = m_isp_container.items[spB_idx.bin_idx][spB_idx.sp_idx];
    
    return !(spB.radius() > 150. && aTriplet.weight < 380.);
}

host_seed_collection atlas_cuts::cutPerMiddleSP(const host_seed_collection& seeds) const {
    
    host_seed_collection newSeeds;
    if (seeds.size() > 1){
	newSeeds.push_back(seeds[0]);
	size_t itLength = std::min(seeds.size(), size_t(5));
	// don't cut first element
	for (size_t i = 1; i < itLength; i++) {
	    if(seeds[i].weight > 200. || seeds[i].spB.radius() > 43. ){
		newSeeds.push_back(std::move(seeds[i]));
	    }
	}
	return newSeeds;	
    }
    return seeds;
}
    
}  // namespace traccc
