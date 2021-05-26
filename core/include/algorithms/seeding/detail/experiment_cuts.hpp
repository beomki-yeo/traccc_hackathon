/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

namespace traccc{

class experiment_cuts{

public:
    virtual ~experiment_cuts() = default;

    virtual void seedWeight(triplet& aTriplet) const = 0;
    
    virtual bool singleSeedCut(const triplet& aTriplet) const = 0;
    
    /// @param seeds contains pairs of weight and seed created for one middle
    /// space
    /// point
    /// @return vector of seeds that pass the cut
    virtual host_seed_collection cutPerMiddleSP(const host_seed_collection& triplets) const = 0;
    
};


} // namespace traccc
