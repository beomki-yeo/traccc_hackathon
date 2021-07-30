/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/Definitions/Common.hpp"
#include "Acts/EventData/TrackParameters.hpp"

// vecmem
#include <vecmem/memory/host_memory_resource.hpp>

// traccc
#include <propagator/propagator.hpp>
#include "edm/collection.hpp"

namespace traccc {
namespace cuda {

template <typename stepper_type, typename navigator_type>
class propagator final {

       
    public:
    template <typename propagator_options_t>
    struct __CUDA_ALIGN__ (16) multi_state {
	
	using stepper_t = stepper_type;
	using navigator_t = navigator_type;
	using propagator_t = typename traccc::propagator< stepper_t, navigator_t >; 
	using state_t = typename propagator_t::template state< propagator_options_t >;
	
        multi_state(const int n_tracks, vecmem::memory_resource* mr)
            : states({typename host_collection< state_t >::item_vector(n_tracks, mr)})

        {}

	/// state vector
	host_collection< state_t > states; 
	
    };

    template <typename state_t, typename surface_t>
    void propagate(state_t& state,
                   host_collection<surface_t>& surfaces,
                   vecmem::memory_resource* resource);
    private:

};

}  // namespace cuda
}  // namespace traccc
