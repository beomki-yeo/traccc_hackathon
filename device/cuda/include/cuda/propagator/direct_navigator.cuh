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

// std
#include <limits>

#include "propagator/direct_navigator.hpp"

namespace traccc {
namespace cuda {

class direct_navigator {

    public:
    using navigator = traccc::direct_navigator;
    using state = typename navigator::state;

    // Wrapper for status call
    template <typename propagator_state_t, typename surface_t>
    static bool status(propagator_state_t& state,
                       host_collection<surface_t>& surfaces,
                       vecmem::memory_resource* resource) {
        return status(state.navigation, state.stepping, surfaces, resource);
    }

    // status call declaration in direct navigator.cu
    template <typename stepper_state_t, typename surface_t>
    static bool status(host_collection<state>& state,
                       host_collection<stepper_state_t>& stepper_state,
                       host_collection<surface_t>& surfaces,
                       vecmem::memory_resource* resource);

    private:
};

}  // namespace cuda
}  // namespace traccc
