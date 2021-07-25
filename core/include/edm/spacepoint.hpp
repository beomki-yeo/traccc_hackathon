/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <definitions/primitives.hpp>
#include <utils/arch_qualifiers.hpp>
#include <vector>

#include "container.hpp"
#include "edm/measurement.hpp"
#include "edm/track_parameters.hpp"
#include "edm/truth/truth_particle.hpp"
#include "geometry/surface.hpp"
// Acts
#include "Acts/EventData/TrackParameters.hpp"

namespace traccc {

/// A spacepoint definition: global position and errors
struct spacepoint {
    point3 global = {0., 0., 0.};
    variance3 variance = {0., 0., 0.};
    geometry_id geom_id;
    Acts::Vector3 truth_mom;
    Acts::ActsScalar time;

    __CUDA_HOST_DEVICE__
    const scalar& x() const { return global[0]; }
    __CUDA_HOST_DEVICE__
    const scalar& y() const { return global[1]; }
    __CUDA_HOST_DEVICE__
    const scalar& z() const { return global[2]; }

    __CUDA_HOST_DEVICE__
    Acts::Vector3 position() const {
        return Acts::Vector3(global[0], global[1], global[2]);
    }

    __CUDA_HOST_DEVICE__
    scalar radius() const {
        return std::sqrt(global[0] * global[0] + global[1] * global[1]);
    }

    // make measurement from spacepoint and surface collection
    // only used for truth tracking
    measurement make_measurement(host_surface_collection& surfaces) {
        const auto& pos = this->position();

        traccc::geometry_id geom_id = this->geom_id;

        // std::cout << geom_id << std::endl;

        // find the surface with the same geometry id
        auto surf_it = std::find_if(
            surfaces.items.begin(), surfaces.items.end(),
            [&geom_id](auto& surf) { return surf.geom_id() == geom_id; });

        // vector indicies of surface
        auto surface_id = std::distance(surfaces.items.begin(), surf_it);

        // Note: loc3[2] should be equal or very close to 0
        Acts::Vector3 loc3 = (*surf_it).global_to_local(pos);

        traccc::point2 loc({loc3[0], loc3[1]});
        traccc::variance2 var({0, 0});

        return measurement({loc, var, surface_id});
    }

    // make measurement from spacepoint and surface collection
    // only used for truth tracking
    bound_track_parameters make_bound_track_parameters(
        host_surface_collection& surfaces, truth_particle& t_particle) {
        const auto& pos = this->position();
        traccc::geometry_id geom_id = this->geom_id;

        // find the surface with the same geometry id
        auto surf_it = std::find_if(
            surfaces.items.begin(), surfaces.items.end(),
            [&geom_id](auto& surf) { return surf.geom_id() == geom_id; });

        // vector indicies of surface
        auto surface_id = std::distance(surfaces.items.begin(), surf_it);

        // Note: loc3[2] should be equal or very close to 0
        Acts::Vector3 loc3 = (*surf_it).global_to_local(pos);

        traccc::point2 loc({loc3[0], loc3[1]});
        traccc::variance2 var({0, 0});

        Acts::Vector3 truth_mom = this->truth_mom;
        Acts::Vector3 truth_mom_dir = truth_mom.normalized();
        Acts::ActsScalar truth_p = truth_mom.norm();	
	
        traccc::bound_track_parameters params;
        params.vector()[Acts::eBoundLoc0] = loc[0];
        params.vector()[Acts::eBoundLoc1] = loc[1];
        params.vector()[Acts::eBoundTime] = this->time;
        params.vector()[Acts::eBoundTheta] =
            Acts::VectorHelpers::theta(truth_mom_dir);
        params.vector()[Acts::eBoundPhi] =
            Acts::VectorHelpers::phi(truth_mom_dir);
        int charge = t_particle.vertex.qop() > 0 ? 1 : -1;
        params.vector()[Acts::eBoundQOverP] = charge / truth_p;
        params.surface_id = surface_id;

        return params;
    }
};

inline bool operator==(const spacepoint& lhs, const spacepoint& rhs) {
    if (std::abs(lhs.global[0] - rhs.global[0]) < 1e-6 &&
        std::abs(lhs.global[1] - rhs.global[1]) < 1e-6 &&
        std::abs(lhs.global[2] - rhs.global[2]) < 1e-6) {
        return true;
    }
    return false;
}

/// Container of spacepoints belonging to one detector module
template <template <typename> class vector_t>
using spacepoint_collection = vector_t<spacepoint>;

/// Convenience declaration for the spacepoint collection type to use in host
/// code
using host_spacepoint_collection = spacepoint_collection<vecmem::vector>;

/// Convenience declaration for the spacepoint collection type to use in device
/// code
using device_spacepoint_collection =
    spacepoint_collection<vecmem::device_vector>;

/// Convenience declaration for the spacepoint container type to use in host
/// code
using host_spacepoint_container = host_container<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container type to use in device
/// code
using device_spacepoint_container = device_container<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container data type to use in
/// host code
using spacepoint_container_data = container_data<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container buffer type to use in
/// host code
using spacepoint_container_buffer = container_buffer<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container view type to use in
/// host code
using spacepoint_container_view = container_view<geometry_id, spacepoint>;

}  // namespace traccc
