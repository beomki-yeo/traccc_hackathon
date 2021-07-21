/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/intersection.hpp"

namespace traccc {

struct stepping_helper {

    template <typename stepper_state_t, typename surface_t>
    static __CUDA_HOST_DEVICE__ intersection::status update_surface_status(
        stepper_state_t& state, surface_t surface) {
        const auto& pos = state.pars.template segment<3>(Acts::eFreePos0);
        const auto& dir = state.pars.template segment<3>(Acts::eFreeDir0);

        auto s_intersection = surface.intersection_estimate(pos, dir);
        if (s_intersection.m_status == intersection::status::on_surface) {
            // release the step size
            state.step_size = 1.;  // need to fix...
            return intersection::status::on_surface;
        } else if (s_intersection.m_status == intersection::status::reachable) {
            Acts::ActsScalar climit = s_intersection.path_length;
            if (climit * climit < state.step_size * state.step_size) {
                state.step_size = state.nav_dir * climit;
            }
            return intersection::status::reachable;
        }
        return intersection::status::unreachable;
    }
};

}  // namespace traccc
