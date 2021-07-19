/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Definitions/Common.hpp"

// std
#include <limits>

namespace traccc {
namespace cuda {

class eigen_stepper{

public:

    template <typename propagator_state_t>
    void step(propagator_state_t& state){


    }   
};


} // namespace cuda    
} // namespace traccc
