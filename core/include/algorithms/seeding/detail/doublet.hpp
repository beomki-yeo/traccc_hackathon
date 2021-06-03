/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once
#include <algorithms/seeding/detail/singlet.hpp>


namespace traccc{
        
    struct lin_circle {
	float Zo;
	float cotTheta;
	float iDeltaR;
	float Er;
	float U;
	float V;
    };

    // Middle - Bottom or Middle - Top
    struct doublet{	
	sp_location sp1;
	sp_location sp2;
	lin_circle lin;
    };

    inline
    bool operator==(const doublet& lhs, const doublet& rhs){
	return (lhs.sp1.bin_idx == rhs.sp1.bin_idx &&
		lhs.sp1.sp_idx == rhs.sp1.sp_idx &&
		lhs.sp2.bin_idx == rhs.sp2.bin_idx &&
		lhs.sp2.sp_idx == rhs.sp2.sp_idx);
    }    
    
    /// Container of doublet belonging to one detector module
    template< template< typename > class vector_t >
    using doublet_collection = vector_t< doublet >;

    /// Convenience declaration for the doublet collection type to use in host code
    using host_doublet_collection
    = doublet_collection< vecmem::vector >;

    /// Convenience declaration for the doublet collection type to use in device code
    using device_doublet_collection
    = doublet_collection< vecmem::device_vector >;

    /// Convenience declaration for the doublet container type to use in host code
    using host_doublet_container
    = host_container< int, doublet >;

    /// Convenience declaration for the doublet container type to use in device code
    using device_doublet_container
    = device_container< int, doublet>;

    /// Convenience declaration for the doublet container data type to use in host code
    using doublet_container_data
    = container_data< int, doublet >;

    /// Convenience declaration for the doublet container buffer type to use in host code
    using doublet_container_buffer
    = container_buffer< int, doublet >;

    /// Convenience declaration for the doublet container view type to use in host code
    using doublet_container_view
    = container_view< int, doublet >;    
    
} // namespace traccc
