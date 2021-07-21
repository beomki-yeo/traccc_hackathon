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

namespace traccc {

template <typename stepper_type, typename navigator_type>
class propagator final {

    public:
    using jacobian_t = Acts::BoundMatrix;
    using stepper_t = stepper_type;
    using stepper_state_t = typename stepper_type::state;
    using navigator_t = navigator_type;
    using navigator_state_t = typename navigator_type::state;
    using surface_t = typename navigator_type::surface_t;

    explicit propagator(stepper_t stepper, navigator_t navigator)
        : m_stepper(std::move(stepper)), m_navigator(std::move(navigator)) {}

    template <typename propagator_options_t>
    struct state {

        state(const propagator_options_t& tops, stepper_state_t stepping_in)
            : options(tops), stepping(stepping_in) {}

        /// These are the options - provided for each propagation step
        propagator_options_t options;

        /// Stepper state - internal state of the Stepper
        stepper_state_t stepping;

        /// Navigation state - internal state of the Navigator
        navigator_state_t navigation;
    };

    template <typename propagator_options_t>
    void propagate(state<propagator_options_t>& state,
                   host_collection<surface_t>& surfaces) {

        // do the eigen stepper
        for (int i_s = 0; i_s < state.options.maxSteps; i_s++) {

            // do navigator
            auto navi_res = navigator_t::status(state, surfaces);

            if (!navi_res) {
                // std::cout << "Total RK steps: " << i_s << std::endl;
                // std::cout << "all targets reached" << std::endl;
                break;
            }

            auto& stepper_state = state.stepping;

            // do RK 4th order
            auto stepper_res = stepper_t::rk4(state);

            if (!stepper_res) {
                std::cout << "stepping failed" << std::endl;
                break;
            }

            // do the covaraince transport
            stepper_t::cov_transport(state);

            // do action for kalman filtering -- currently empty
            state.options.action(state, m_stepper);
        }
    }

    // not used
    template <typename parameters_t, typename propagator_options_t>
    void propagate(const parameters_t& start,
                   const propagator_options_t& options) const {}

    private:
    stepper_t m_stepper;
    navigator_t m_navigator;
};

}  // namespace traccc
