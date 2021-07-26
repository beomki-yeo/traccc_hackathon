/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/Definitions/Algebra.hpp"

namespace traccc {

struct intersection {

    enum class status : int {
        missed = 0,
        unreachable = 0,
        reachable = 1,
        on_surface = 2
    };

    /// Position of the intersection
    Acts::Vector3 position{0., 0., 0.};
    /// Signed path length to the intersection (if valid)
    Acts::ActsScalar path_length{
        std::numeric_limits<Acts::ActsScalar>::infinity()};
    /// The Status of the intersection
    status m_status{status::unreachable};

    __CUDA_HOST_DEVICE__
    intersection(const Acts::Vector3 &sinter, Acts::ActsScalar slength,
                 status sstatus)
        : position(sinter), path_length(slength), m_status(sstatus) {}

    intersection() = default;
};

}  // namespace traccc
