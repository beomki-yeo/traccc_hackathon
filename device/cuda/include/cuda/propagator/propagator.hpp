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

namespace traccc {    
namespace cuda {
    
template <typename stepper_t, typename navigator_t >
class propagator final {

public:
    //using jacobian_t = Acts::BoundMatrix;
    //using stepper_t = stepper_type;
    //using navigator_t = navigator_type;

    using stepper_state_t = typename stepper_t::state;
    using navigator_state_t = typename navigator_t::state;
    
    explicit propagator(stepper_t stepper, navigator_t navigator):
	m_stepper(std::move(stepper)), m_navigator(std::move(navigator)){}
    
    template < typename propagator_options_t >
    struct state {

	
	template < typename params_container_t >
	state(const params_container_t& start,
	      const propagator_options_t& tops,
	      vecmem::memory_resource* mr):
	    options(tops),
	    stepping({typename host_collection<stepper_state_t>::item_vector(start.size(), mr)}),
	    navigation({typename host_collection<navigator_state_t>::item_vector(start.size(), mr)})
	{

	}

	/// These are the options - provided for each propagation step
	propagator_options_t options;

	/// Stepper state container - internal states of the Stepper
	host_collection<stepper_state_t> stepping;

	/// Navigator state container - internal states of the Navigator
	host_collection<navigator_state_t> navigation;	
    };

    template < typename parameters_container_t, typename propagator_options_t  >
    void propagate(const parameters_container_t& start,
		   const propagator_options_t& options) const{

	// define state container type
	using state_t = state<propagator_options_t>;

	state_t state{start, options,
		      m_stepper.make_state(start, options.maxStepSize, options.tolerance)};		

	// Start propagation	
	m_navigator.target(state, m_stepper);	
	for (size_t i_s=0; i_s < state.options.maxSteps; ++i_s) {
	    
	    m_stepper.step(state);
	    
	    m_navigator.status(state, m_stepper);
	    
	    state.options.action(state, m_stepper);

	    if (state.options.abort(state, m_stepper)) {
		break;
	    }
	    
	    m_navigator.target(state, m_stepper);	
	}
    }    
    
private:
    stepper_t m_stepper;
    navigator_t m_navigator;
};

}  // namespace cuda
}  // namespace traccc
