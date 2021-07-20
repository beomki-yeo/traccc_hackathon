/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <definitions/algebra.hpp>

#include "edm/collection.hpp"
#include "utils/arch_qualifiers.hpp"

// Acts
#include "Acts/EventData/TrackParameters.hpp"

namespace traccc {

class surface {

    public:
    surface() = default;
    
    // construct surface based on center and normal vector
    surface(const Acts::Vector3& center, const Acts::Vector3& normal,
            geometry_id geom_id)
        : m_geom_id(geom_id) {
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
    }

    // construct surface based on transform3
    surface(const traccc::transform3& tf, const traccc::geometry_id& geom_id){
	Acts::Vector3 e_normal;
	e_normal(0,0) = tf._data[2][0];
	e_normal(1,0) = tf._data[2][1];
	e_normal(2,0) = tf._data[2][2];

	Acts::Vector3 e_center;
	e_center(0,0) = tf._data[3][0];
	e_center(1,0) = tf._data[3][1];
	e_center(2,0) = tf._data[3][2];
	
	surface(e_center, e_normal, geom_id);	
    }
    
    __CUDA_HOST_DEVICE__
    Acts::Transform3 transform() { return m_transform; }

    __CUDA_HOST_DEVICE__
    geometry_id geom_id() { return m_geom_id; }

    __CUDA_HOST_DEVICE__
    Acts::Vector3 local_to_global(const Acts::Vector2& loc) {
        return m_transform *
               Acts::Vector3(loc[Acts::eBoundLoc0], loc[Acts::eBoundLoc1], 0.);
    }

    __CUDA_HOST_DEVICE__
    Acts::Vector3 global_to_local(const Acts::Vector3& glo) {
        return m_transform.inverse() * glo;
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
