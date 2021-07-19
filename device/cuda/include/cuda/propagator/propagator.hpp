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
namespace cuda {
    
template <typename stepper_type, typename navigator_type >
class propagator final {

public:
    using jacobian_t = Acts::BoundMatrix;
    using stepper_t = stepper_type;
    using stepper_state_container_t = typename stepper_type::state_container;
    using navigator_t = navigator_type;
    using navigator_state_container_t = typename navigator_type::state_container;
    
    template < typename propagator_options_t >
    struct state {
	
	template < typename parameters_container_t >
	state(const parameters_container_t& start, // supposed to be on device
	      const propagator_options_t& tops):
	    options(tops){

	}

	/// These are the options - provided for each propagation step
	propagator_options_t options;

	/// Stepper state container - internal states of the Stepper
	stepper_state_container_t stepping;

	/// Navigator state container - internal states of the Navigator
	navigator_state_container_t navigator;
	
    };

    template < typename parameters_container_t, typename propagator_options_t  >
    void propagate(const parameters_container_t& start, // supposed to be on device
		   const propagator_options_t& options) const{

	using state_t = state<propagator_options_t>;

	// do it on GPU?
	state_t state{start, options,
		      m_stepper.make_state(start, options.maxStepSize, options.tolerance)};		

	// do it on GPU
	m_stepper(state);
	
    }    

    
private:
    stepper_t m_stepper;    
};

}  // namespace cuda
}  // namespace traccc
