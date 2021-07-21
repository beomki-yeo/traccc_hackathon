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

template <typename surface_type = surface>
class direct_navigator {
    public:
    using surface_t = surface_type;

    struct state {

        // surface sequence: takes surface id as input
        array<size_t, 30> surface_sequence;

        /// The surface sequence size
        size_t surface_sequence_size = 0;

        // surface iterator id
        size_t surface_iterator_id = 0;

        /// Navigation state - external interface: the start surface
        size_t start_surface_id = 0;

        /// Navigation state - external interface: the current surface
        size_t current_surface_id = 0;

        /// Navigation state - external interface: the target surface
        size_t target_surface_id = 0;

        /// Navigation state - external interface: target is reached
        bool target_reached = false;

        /// Navigation state - external interface: a break has been detected
        bool navigation_break = false;
    };

    template <typename propagator_state_t>
    static bool status(propagator_state_t& state,
                       host_collection<surface_t>& surfaces) {
        return status(state.navigation, state.stepping, surfaces);
    }

    template <typename stepper_t>
    static __CUDA_HOST_DEVICE__ bool status(
        state& state, stepper_t& stepper_state,
        host_collection<surface_t>& surfaces) {

	if (state.surface_iterator_id >= state.surface_sequence_size){
	    return false;
	}
	
        // check if we are on surface
        if (state.surface_iterator_id < state.surface_sequence_size) {

            // set first target surface id
            if (state.surface_iterator_id == 0) {
                state.target_surface_id = state.surface_sequence[0];
            }

            surface_t& target_surface = surfaces.items[state.target_surface_id];

            // establish the surface status
            auto surface_status = stepping_helper::update_surface_status(
                stepper_state, target_surface);

            // if the stepper state is on surface
            if (surface_status == intersection::status::on_surface) {

                // increase the iterator id
                state.surface_iterator_id++;

                // update the target surface id
                state.target_surface_id =
                    state.surface_sequence[state.surface_iterator_id];
            }
        }

	return true;
    }

    template <typename propagator_state_t, typename stepper_t>
    static void target(propagator_state_t& state, const stepper_t& stepper) {
        target(state.navigation, stepper);
    }

    template <typename stepper_t>
    static __CUDA_HOST_DEVICE__ void target(state& state,
                                            const stepper_t& stepper) {}
};

}  // namespace traccc
