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

    template <typename propagator_state_t, typename surface_t>
    static void status(propagator_state_t& state,
                       host_collection<surface_t>& surfaces);

    private:
};

}  // namespace cuda
}  // namespace traccc
