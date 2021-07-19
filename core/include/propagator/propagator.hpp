/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Acts
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Definitions/Common.hpp"

namespace traccc {    
    
template < typename stepper_type, typename navigator_type >
class propagator final {

public:
    using jacobian_t = Acts::BoundMatrix;
    using stepper_t = stepper_type;
    using stepper_state_t = typename stepper_type::state;
    using navigator_t = navigator_type;
    using navigator_state_t = typename navigator_type::state;
    
    template < typename propagator_options_t >
    struct state {
	
	template < typename parameters_t >
	state(const parameters_t& start,
	      const propagator_options_t& tops,
	      stepper_state_t stepping_in):
	    options(tops),
	    stepping(std::move(stepping_in)){

	}

	/// These are the options - provided for each propagation step
	propagator_options_t options;

	/// Stepper state - internal state of the Stepper
	stepper_state_t stepping;

	/// Navigation state - internal state of the Navigator
	navigator_state_t navigation;
    };

    template < typename parameters_t, typename propagator_options_t  >
    void propagate(const parameters_t& start,
		   const propagator_options_t& options) const{

	using state_t = state<propagator_options_t>;
	
	state_t state{start, options,
		      m_stepper.make_state(start, options.maxStepSize, options.tolerance)};		

	for (unsigned int i_s=0; i_s < state.options.maxSteps; ++i_s) {

	    m_stepper.step(state);

	}	
    }    

private:
    stepper_t m_stepper;
};

}  // namespace traccc
