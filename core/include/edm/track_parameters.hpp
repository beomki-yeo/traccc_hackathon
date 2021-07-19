/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "edm/collection.hpp"
#include "geometry/surface.hpp"
#include "utils/arch_qualifiers.hpp"

// Acts
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"

namespace traccc {

struct bound_track_parameters {
    using vector_t = Acts::BoundVector;
    using covariance_t = Acts::BoundSymMatrix;
    using jacobian_t = Acts::BoundMatrix;

    vector_t m_vector;
    covariance_t m_covariance;
    std::shared_ptr<surface> m_surface;

    __CUDA_HOST_DEVICE__
    auto charge() const {
        if (m_vector[Acts::eBoundQOverP] < 0) {
            return -1.;
        } else {
            return 1.;
        }
    }

    __CUDA_HOST_DEVICE__
    auto& vector() { return m_vector; }

    __CUDA_HOST_DEVICE__
    auto& covariance() { return m_covariance; }
};

struct curvilinear_track_parameters {
    using vector_t = Acts::BoundVector;
    using covariance_t = Acts::BoundSymMatrix;
    using jacobian_t = Acts::BoundMatrix;

    vector_t m_vector;
    covariance_t m_covariance;

    __CUDA_HOST_DEVICE__
    auto& vector() { return m_vector; }

    __CUDA_HOST_DEVICE__
    auto& covariance() { return m_covariance; }
};

struct free_track_parameters {

    using vector_t = Acts::FreeVector;
    using covariance_t = Acts::FreeSymMatrix;
    using jacobian_t = Acts::FreeMatrix;

    vector_t m_vector;
    covariance_t m_covariance;

    __CUDA_HOST_DEVICE__
    auto& vector() { return m_vector; }

    __CUDA_HOST_DEVICE__
    auto& covariance() { return m_covariance; }
};

};  // namespace traccc
