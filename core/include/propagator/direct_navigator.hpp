/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "geometry/surface.hpp"
#include "propagator/stepping_helper.hpp"

namespace traccc {

class direct_navigator {
    public:
    struct __CUDA_ALIGN__(16) state {

        /// Navigation state - external interface: target is reached
        int target_reached = false;

        /// Navigation state - external interface: a break has been detected
        int navigation_break = false;

        // surface sequence: takes surface id as input
        std::array<unsigned int, 30> surface_sequence;

        /// The surface sequence size
        unsigned int surface_sequence_size = 0;

        // surface iterator id
        unsigned int surface_iterator_id = 0;

        /// Navigation state - external interface: the start surface
        unsigned int start_surface_id = 0;

        /// Navigation state - external interface: the current surface
        unsigned int current_surface_id = 0;

        /// Navigation state - external interface: the target surface
        unsigned int target_surface_id = 0;
    };

    template <typename propagator_state_t, typename surface_t>
    static bool status(propagator_state_t& state, surface_t* surfaces) {
        return status(state.navigation, state.stepping, surfaces);
    }

    template <typename stepper_state_t, typename surface_t>
    static __CUDA_HOST_DEVICE__ bool status(state& state,
                                            stepper_state_t& stepper_state,
                                            surface_t* surfaces) {

        if (state.surface_iterator_id >= state.surface_sequence_size) {
            return false;
        }

        // check if we are on surface
        if (state.surface_iterator_id < state.surface_sequence_size) {

            // set first target surface id
            if (state.surface_iterator_id == 0) {
                state.target_surface_id = state.surface_sequence[0];
            }

            surface_t* target_surface = surfaces + state.target_surface_id;

            // establish the surface status
            auto surface_status = stepping_helper::update_surface_status(
                stepper_state, target_surface);

            // if the stepper state is on surface
            if (surface_status == intersection::status::on_surface) {

                // printf("on surface %u %u \n", state.surface_iterator_id,
                // state.surface_sequence_size);

                // increase the iterator id
                state.surface_iterator_id++;

                // update the target surface id
                state.target_surface_id =
                    state.surface_sequence[state.surface_iterator_id];
            }
        }

        return true;
    }
};

}  // namespace traccc
