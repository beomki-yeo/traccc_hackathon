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
    explicit __CUDA_HOST_DEVICE__ propagator(stepper_type stepper,
                                             navigator_type navigator)
        : m_stepper(std::move(stepper)), m_navigator(std::move(navigator)) {}

    template <typename propagator_options_t>
    struct __CUDA_ALIGN__(16) state {

        using jacobian_t = Acts::BoundMatrix;
        using stepper_t = stepper_type;
        using stepper_state_t = typename stepper_type::state;
        using navigator_t = navigator_type;
        using navigator_state_t = typename navigator_type::state;

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
    __CUDA_HOST_DEVICE__ void propagate(state_t& state, surface_t* surfaces) {

        // Pre-stepping

        // navigator initialize state call
        m_navigator.status(state, surfaces);

        // do action for kalman filtering -- currently empty
        // state.options.action(state, m_stepper);

        if (!m_navigator.target(state, surfaces)) {
            return;
        }

        for (unsigned int i_s = 0; i_s < state.options.maxSteps; i_s++) {

            auto stepper_res = m_stepper.step(state);

            if (!stepper_res) {
                break;
            }

            // Post-stepping
            m_navigator.status(state, surfaces);

            // do action for kalman filtering -- currently empty
            // state.options.action(state, m_stepper);

            auto navi_res = m_navigator.target(state, surfaces);

            if (!navi_res) {
                break;
            }
        }
    }

    private:
    stepper_type m_stepper;
    navigator_type m_navigator;
};

}  // namespace traccc
