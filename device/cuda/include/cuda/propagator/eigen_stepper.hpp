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

namespace traccc {
namespace cuda {

class eigen_stepper{
    
public:

    using state = traccc::eigen_stepper::state;    
    using host_state_collection = host_collection<state>;
    using device_state_collection = device_collection<state>;
    using state_collection_data = collection_data<state>;
    using state_collection_buffer = collection_buffer<state>;
    using state_collection_view = collection_view<state>;
    
    template <typename propagator_state_t>
    void step(propagator_state_t& state){


    }   
};


} // namespace cuda    
} // namespace traccc
