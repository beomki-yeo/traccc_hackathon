/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/arch_qualifiers.hpp"
#include <definitions/algebra.hpp>

// Acts
#include "Acts/EventData/TrackParameters.hpp"


namespace traccc {

class surface {

public:
    surface(const transform3& transform) {
	//m_transform = transform;
    }
    
    __CUDA_HOST_DEVICE__
    Acts::Transform3 transform() { return m_transform; }
    
    __CUDA_HOST_DEVICE__
    Acts::Vector3 local_to_global (const Acts::Vector2& loc){
	return m_transform * Acts::Vector3(loc[Acts::eBoundLoc0], loc[Acts::eBoundLoc1], 0.);	
    }
    
private:
    //transform3 m_transform;
    Acts::Transform3 m_transform;
};

/// Convenience declaration for the surface collection type to use in host
/// code
using host_surface_collection = host_collection<surface>;

/// Convenience declaration for the surface collection type to use in device
/// code
using device_surface_collection = device_collection<surface>;

using surface_collection_data = collection_data<surface>;

using surface_collection_buffer = collection_buffer<surface>;

using surface_collection_view = collection_view<surface>;
    
}  // namespace traccc
