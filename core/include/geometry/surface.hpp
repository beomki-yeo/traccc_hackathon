/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/TrackParameters.hpp"

// traccc
#include <definitions/algebra.hpp>

#include "edm/collection.hpp"
#include "edm/detail/transform_bound_to_free.hpp"
#include "utils/arch_qualifiers.hpp"
#include "utils/intersection.hpp"
#include "utils/vector_helpers.hpp"

namespace traccc {

class surface {

    public:
    surface() = default;

    // construct surface based on transform3
    surface(const traccc::transform3& tf, geometry_id geom_id) {
        Acts::Vector3 normal;
        normal(0, 0) = tf._data[2][0];
        normal(1, 0) = tf._data[2][1];
        normal(2, 0) = tf._data[2][2];

        Acts::Vector3 center;
        center(0, 0) = tf._data[3][0];
        center(1, 0) = tf._data[3][1];
        center(2, 0) = tf._data[3][2];

        // surface(e_center, e_normal, geom_id);

        Acts::Vector3 T = normal.normalized();
        Acts::Vector3 U = std::abs(T.dot(Acts::Vector3::UnitZ())) <
                                  Acts::s_curvilinearProjTolerance
                              ? Acts::Vector3::UnitZ().cross(T).normalized()
                              : Acts::Vector3::UnitX().cross(T).normalized();
        Acts::Vector3 V = T.cross(U);
        Acts::RotationMatrix3 curvilinearRotation;
        curvilinearRotation.col(0) = U;
        curvilinearRotation.col(1) = V;
        curvilinearRotation.col(2) = T;

        // curvilinear surfaces are boundless
        m_transform = Acts::Transform3{curvilinearRotation};
        m_transform.pretranslate(center);

        m_geom_id = geom_id;
    }

    __CUDA_HOST_DEVICE__
    Acts::Transform3 transform() { return m_transform; }

    __CUDA_HOST_DEVICE__
    geometry_id geom_id() { return m_geom_id; }

    __CUDA_HOST_DEVICE__
    Acts::Vector3 local_to_global(const Acts::Vector2& loc) const {
        return m_transform *
               Acts::Vector3(loc[Acts::eBoundLoc0], loc[Acts::eBoundLoc1], 0.);
    }

    __CUDA_HOST_DEVICE__
    Acts::Vector3 global_to_local(const Acts::Vector3& glo) {
        return m_transform.inverse() * glo;
    }

    __CUDA_HOST_DEVICE__
    Acts::Vector3 global_to_local(const Acts::Vector3& glo) const {
        return m_transform.inverse() * glo;
    }

    __CUDA_HOST_DEVICE__
    Acts::RotationMatrix3 reference_frame() const {
        return m_transform.matrix().block<3, 3>(0, 0);
    }

    __CUDA_HOST_DEVICE__
    Acts::FreeToBoundMatrix free_to_bound_jacobian(
        const Acts::FreeVector& free_vector) const {
        // The global position
        const auto position = free_vector.segment<3>(Acts::eFreePos0);
        // The direction
        const auto direction = free_vector.segment<3>(Acts::eFreeDir0);
        // Use fast evaluation function of sin/cos
        auto [cosPhi, sinPhi, cosTheta, sinTheta, invSinTheta] =
            vector_helpers::evaluate_trigonomics(direction);

        // The measurement frame of the surface
        const auto& rframeT = reference_frame().transpose();

        // Initalize the jacobian from global to local
        Acts::FreeToBoundMatrix jac_to_local = Acts::FreeToBoundMatrix::Zero();
        // Local position component given by the refernece frame
        jac_to_local.block<2, 3>(Acts::eBoundLoc0, Acts::eFreePos0) =
            rframeT.block<2, 3>(0, 0);
        // Time component
        jac_to_local(Acts::eBoundTime, Acts::eFreeTime) = 1;
        // Directional and momentum elements for reference frame surface
        jac_to_local(Acts::eBoundPhi, Acts::eFreeDir0) = -sinPhi * invSinTheta;
        jac_to_local(Acts::eBoundPhi, Acts::eFreeDir1) = cosPhi * invSinTheta;
        jac_to_local(Acts::eBoundTheta, Acts::eFreeDir0) = cosPhi * cosTheta;
        jac_to_local(Acts::eBoundTheta, Acts::eFreeDir1) = sinPhi * cosTheta;
        jac_to_local(Acts::eBoundTheta, Acts::eFreeDir2) = -sinTheta;
        jac_to_local(Acts::eBoundQOverP, Acts::eFreeQOverP) = 1;

        return jac_to_local;
    }

    __CUDA_HOST_DEVICE__
    Acts::FreeToPathMatrix free_to_path_derivative(
        const Acts::FreeVector& free_vector) const {
        // The global position
        const auto& position = free_vector.segment<3>(Acts::eFreePos0);
        // The direction
        const auto& direction = free_vector.segment<3>(Acts::eFreeDir0);
        // The measurement frame of the surface
        const Acts::RotationMatrix3& rframe = reference_frame();

        // The measurement frame z axis
        const Acts::Vector3 refZAxis = rframe.col(2);
        // Cosine of angle between momentum direction and measurement frame z
        // axis
        const Acts::ActsScalar dz = refZAxis.dot(direction);
        // Initialize the derivative
        Acts::FreeToPathMatrix free_to_path = Acts::FreeToPathMatrix::Zero();
        free_to_path.segment<3>(Acts::eFreePos0) =
            -1.0 * refZAxis.transpose() / dz;
        return free_to_path;
    }

    __CUDA_HOST_DEVICE__
    Acts::BoundToFreeMatrix bound_to_free_jacobian(
        const Acts::BoundVector& bound_vector) const {
        // Transform from bound to free parameters
        Acts::FreeVector free_vector =
            detail::transform_bound_to_free_parameters(bound_vector, *this);

        // The global position
        const Acts::Vector3 position = free_vector.segment<3>(Acts::eFreePos0);
        // The direction
        const Acts::Vector3 direction = free_vector.segment<3>(Acts::eFreeDir0);
        // Use fast evaluation function of sin/cos
        auto [cosPhi, sinPhi, cosTheta, sinTheta, invSinTheta] =
            vector_helpers::evaluate_trigonomics(direction);

        // retrieve the reference frame
        const Acts::RotationMatrix3& rframe = reference_frame();
        // Initialize the jacobian from local to global
        Acts::BoundToFreeMatrix jac_to_global = Acts::BoundToFreeMatrix::Zero();
        // the local error components - given by reference frame
        jac_to_global.topLeftCorner<3, 2>() = rframe.topLeftCorner<3, 2>();
        // the time component
        jac_to_global(Acts::eFreeTime, Acts::eBoundTime) = 1;
        // the momentum components
        jac_to_global(Acts::eFreeDir0, Acts::eBoundPhi) = (-sinTheta) * sinPhi;
        jac_to_global(Acts::eFreeDir0, Acts::eBoundTheta) = cosTheta * cosPhi;
        jac_to_global(Acts::eFreeDir1, Acts::eBoundPhi) = sinTheta * cosPhi;
        jac_to_global(Acts::eFreeDir1, Acts::eBoundTheta) = cosTheta * sinPhi;
        jac_to_global(Acts::eFreeDir2, Acts::eBoundTheta) = (-sinTheta);
        jac_to_global(Acts::eFreeQOverP, Acts::eBoundQOverP) = 1;
        return jac_to_global;
    }

    __CUDA_HOST_DEVICE__
    intersection intersection_estimate(const Acts::Vector3& position,
                                       const Acts::Vector3& direction) {

        // Get the matrix from the transform (faster access)
        const auto& t_matrix = m_transform.matrix();
        const Acts::Vector3& pnormal = t_matrix.block<3, 1>(0, 2).transpose();
        const Acts::Vector3& pcenter = t_matrix.block<3, 1>(0, 3).transpose();
        // It is solvable, so go on
        Acts::ActsScalar denom = direction.dot(pnormal);
        if (denom != 0.0) {
            // Translate that into a path
            Acts::ActsScalar path =
                (pnormal.dot((pcenter - position))) / (denom);
            // Is valid hence either on surface or reachable
            intersection::status status =
                (path * path <
                 Acts::s_onSurfaceTolerance * Acts::s_onSurfaceTolerance)
                    ? intersection::status::on_surface
                    : intersection::status::reachable;
            // Return the intersection
            return intersection{(position + path * direction), path, status};
        }

        return intersection();
    }

    private:
    Acts::Transform3 m_transform;
    geometry_id m_geom_id;
};

/// Convenience declaration for the surface collection type to use in host
/// code
using host_surface_collection = host_collection<surface>;

/// Convenience declaration for the surface collection type to use in device
/// code
using device_surface_collection = device_collection<surface>;

using surface_collection_data = collection_data<surface>;

using surface_collection_buffer = collection_buffer<surface>;

using surface_collection_view = collection_view<surface>;

}  // namespace traccc
