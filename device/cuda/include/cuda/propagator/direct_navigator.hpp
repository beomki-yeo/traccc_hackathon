/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Definitions/Common.hpp"

// std
#include <limits>

#include "propagator/direct_navigator.hpp"

namespace traccc {
namespace cuda {

template < typename surface_type = surface >
class direct_navigator{
    
public:

    using surface_t = surface_type;

    using state = typename traccc::direct_navigator<surface_t>::state;    
    using host_state_collection = host_collection<state>;
    using device_state_collection = device_collection<state>;
    using state_collection_data = collection_data<state>;
    using state_collection_buffer = collection_buffer<state>;
    using state_collection_view = collection_view<state>;
    
private:

};    
   
} // namespace cuda    
} // namespace traccc
    
