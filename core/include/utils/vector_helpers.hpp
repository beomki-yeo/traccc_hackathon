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
namespace vector_helpers {

static __CUDA_HOST_DEVICE__ std::array<Acts::ActsScalar, 5>
evaluate_trigonomics(const Acts::Vector3& direction) {

    const Acts::ActsScalar x = direction(0);  // == cos(phi) * sin(theta)
    const Acts::ActsScalar y = direction(1);  // == sin(phi) * sin(theta)
    const Acts::ActsScalar z = direction(2);  // == cos(theta)
    // can be turned into cosine/sine
    const Acts::ActsScalar cosTheta = z;
    const Acts::ActsScalar sinTheta = std::sqrt(x * x + y * y);
    const Acts::ActsScalar invSinTheta = 1. / sinTheta;
    const Acts::ActsScalar cosPhi = x * invSinTheta;
    const Acts::ActsScalar sinPhi = y * invSinTheta;

    return {cosPhi, sinPhi, cosTheta, sinTheta, invSinTheta};
}

static __CUDA_HOST_DEVICE__ Acts::ActsMatrix<3, 3> cross(
    const Acts::ActsMatrix<3, 3>& m, const Acts::Vector3& v) {
    Acts::ActsMatrix<3, 3> r;
    r.col(0) = m.col(0).cross(v);
    r.col(1) = m.col(1).cross(v);
    r.col(2) = m.col(2).cross(v);

    return r;
}

}  // namespace vector_helpers
}  // namespace traccc
