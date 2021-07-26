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

// traccc
#include <utils/arch_qualifiers.hpp>

namespace traccc {

template <typename stepper_type, typename navigator_type>
class propagator final {

    public:
    using jacobian_t = Acts::BoundMatrix;
    using stepper_t = stepper_type;
    using stepper_state_t = typename stepper_type::state;
    using navigator_t = navigator_type;
    using navigator_state_t = typename navigator_type::state;

    explicit __CUDA_HOST_DEVICE__ propagator(stepper_t stepper,
                                             navigator_t navigator)
        : m_stepper(std::move(stepper)), m_navigator(std::move(navigator)) {}

    template <typename propagator_options_t>
    struct __CUDA_ALIGN__(16) state {

        state() = default;

        state(const propagator_options_t& tops, stepper_state_t stepping_in)
            : options(tops), stepping(stepping_in) {}

        /// These are the options - provided for each propagation step
        propagator_options_t options;

        /// Stepper state - internal state of the Stepper
        stepper_state_t stepping;

        /// Navigation state - internal state of the Navigator
        navigator_state_t navigation;
    };

    template <typename state_t, typename surface_t>
    void propagate(state_t& state, surface_t* surfaces) {
        propagate(state.options, state.stepping, state.navigation, surfaces);
    }

    template <typename propagator_options_t, typename stepper_state_t,
              typename navigation_state_t, typename surface_t>
    __CUDA_HOST_DEVICE__ void propagate(propagator_options_t& options,
                                        stepper_state_t& stepping,
                                        navigation_state_t& navigation,
                                        surface_t* surfaces) {
        // do the eigen stepper
        for (int i_s = 0; i_s < options.maxSteps; i_s++) {

            // do navigator
            auto navi_res = navigator_t::status(navigation, stepping, surfaces);

            if (!navi_res) {
                // printf("Total RK steps: %d \n", i_s);
                // printf("all targets reached \n");
                break;
            }

            // do RK 4th order
            auto stepper_res = stepper_t::rk4(stepping);

            if (!stepper_res) {
                printf("stepper break \n");
                break;
            }

            // do the covaraince transport
            stepper_t::cov_transport(stepping, options.mass);

            // do action for kalman filtering -- currently empty
            // state.options.action(state, m_stepper);
        }
    }

    private:
    stepper_t m_stepper;
    navigator_t m_navigator;
};

}  // namespace traccc
