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
    
    template <typename propagator_state_t>
    void step(propagator_state_t& state){
	// left empty for the moment
    }

    // Wrapper for rk4
    template <typename propagator_state_t>
    static bool rk4(propagator_state_t& state){
	return rk4(state.stepping);
    }

    // rk4 declaration in eigen_stepper.cu
    static bool rk4(host_collection<state>& state);

    // Wrapper for cov transport
    template <typename propagator_state_t>
    static void cov_transport(propagator_state_t& state){
	cov_transport(state.stepping, state.options.mass);
    }

    // cov transport declaration in eigen_stepper.cu
    static void cov_transport(host_collection<state>& state, const Acts::ActsScalar mass);
    
private:
    
};


} // namespace cuda    
} // namespace traccc
