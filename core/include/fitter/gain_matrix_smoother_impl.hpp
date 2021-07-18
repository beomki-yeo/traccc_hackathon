/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

template <typename track_state_t>
struct gain_matrix_smoother_impl {

    static __CUDA_HOST_DEVICE__ void smoother(track_state_t& tr_state) {

        // Fill out the algorithm
    }
};

}  // namespace traccc
