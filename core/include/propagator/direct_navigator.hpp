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
    static __CUDA_HOST_DEVICE__ void status(propagator_state_t& state,
                                            surface_t* surfaces) {
        /*
        if (state.navigation.surface_iterator_id >=
            state.navigation.surface_sequence_size) {
            return false;
        }
        */
        // check if we are on surface
        if (state.navigation.surface_iterator_id <
            state.navigation.surface_sequence_size) {

            // set first target surface id
            if (state.navigation.surface_iterator_id == 0) {
                state.navigation.target_surface_id =
                    state.navigation.surface_sequence[0];
            }

            surface_t* target_surface =
                surfaces + state.navigation.target_surface_id;

            // establish the surface status
            auto surface_status = stepping_helper::update_surface_status(
                state.stepping, target_surface);

            // if the stepper state is on surface
            if (surface_status == intersection::status::on_surface) {

                // increase the iterator id
                state.navigation.surface_iterator_id++;

                // update the target surface id
                state.navigation.target_surface_id =
                    state.navigation
                        .surface_sequence[state.navigation.surface_iterator_id];
            }
        }

        // return true;
    }

    template <typename propagator_state_t, typename surface_t>
    __CUDA_HOST_DEVICE__ bool target(propagator_state_t& state,
                                     surface_t* surfaces) {

        // check if there is more surfaces to pass through
        if (state.navigation.surface_iterator_id <
            state.navigation.surface_sequence_size) {

            surface_t* target_surface =
                surfaces + state.navigation.target_surface_id;

            // establish the surface status
            auto surface_status = stepping_helper::update_surface_status(
                state.stepping, target_surface);

            if (surface_status == intersection::status::unreachable) {
                state.navigation.surface_iterator_id++;

                // update the target surface id
                state.navigation.target_surface_id =
                    state.navigation
                        .surface_sequence[state.navigation.surface_iterator_id];
            }
            return true;
        }
        // otherwise break
        else {
            // current not used
            state.navigation.navigation_break = true;
            // currently just use boolean result
            return false;
        }
    }
};

}  // namespace traccc
