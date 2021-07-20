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
#include <propagator/eigen_stepper.hpp>

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

    template <typename propagator_state_t>
    static void cov_transport(propagator_state_t& state){
	cov_transport(state.stepping, state.options.mass);
    }

    static void cov_transport(host_state_collection& state, const Acts::ActsScalar mass);
    
private:
    
};


} // namespace cuda    
} // namespace traccc
