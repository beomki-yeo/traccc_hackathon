/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <geometry/surface.hpp>

namespace traccc {

template <typename propagator_t, typename updater_t, typename smoother_t>
class kalman_fitter {

    using kalman_navigator_t = typename propagator_t::navigator_t;

    public:
    kalman_fitter(propagator_t pPropagator)
        : m_propagator(std::move(pPropagator)) {}

    private:
    propagator_t m_propagator;

    template <typename parameters_t>
    class actor {
        public:
        // ID of target surface
        int target_surface_id;

        /// Whether to consider multiple scattering.
        bool multiple_scattering = true;

        /// Whether to consider energy loss.
        bool energy_loss = true;

        /// Whether run reversed filtering
        bool reversed_filtering = false;

        template <typename propagator_state_t, typename stepper_t>
        void operator()(propagator_state_t& state,
                        const stepper_t& stepper) const {}
    };

    template <typename parameters_t>
    class aborter {};
};

}  // namespace traccc
