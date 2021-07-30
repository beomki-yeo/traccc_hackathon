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

namespace traccc {
namespace cuda {

class eigen_stepper {

    public:
    template <typename propagator_state_t>
    void step(propagator_state_t& state) {
        // left empty for the moment
    }

    template <typename propagator_state_t>
    static void rk4(propagator_state_t& state);

    template <typename propagator_state_t>
    static void cov_transport(propagator_state_t& state);

    private:
};

}  // namespace cuda
}  // namespace traccc
