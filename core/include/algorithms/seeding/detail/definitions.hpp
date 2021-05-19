/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

namespace traccc{
    
    struct sp_location{
	/// index of the bin of the spacepoint grid
	size_t bin_idx;
	/// index of the spacepoint in the bin
	size_t sp_idx; 
    };

    struct lin_circle {
	float Zo;
	float cotTheta;
	float iDeltaR;
	float Er;
	float U;
	float V;
    };    

} // namespace traccc
