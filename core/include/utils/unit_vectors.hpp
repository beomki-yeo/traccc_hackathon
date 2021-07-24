/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "Acts/Definitions/Algebra.hpp"

namespace traccc {

template <typename T>
static __CUDA_HOST_DEVICE__ Eigen::Matrix<T, 3, 1>
make_direction_unit_from_phi_theta(T phi, T eta) {
    const auto coshEtaInv = 1 / std::cosh(eta);
    return {
        std::cos(phi) * coshEtaInv,
        std::sin(phi) * coshEtaInv,
        std::tanh(eta),
    };
}

}  // namespace traccc
