/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

// Some useful helper functions for cuda device
struct cuda_helper {

    /// reduce sum function to obtain the sum of elements in array
    ///
    /// @param array the input array    
    template <typename T>
    static __device__ void reduce_sum(T* array) {
	const auto& tid = threadIdx.x;
        array[tid] +=
            __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 2, WARP_SIZE);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 4,
                                       WARP_SIZE / 2);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 8,
                                       WARP_SIZE / 4);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 16,
                                       WARP_SIZE / 8);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 32,
                                       WARP_SIZE / 16);

        __syncthreads();

        if (tid == 0) {
            for (int i = 1; i < blockDim.x / WARP_SIZE; i++) {
                array[tid] += array[i * WARP_SIZE];
            }
        }
    }

    template <typename T>
    static __device__ void get_bin_idx(
        const unsigned int& n_bins,
        const vecmem::jagged_device_vector<T>& jag_vec, unsigned int& bin_idx,
        unsigned int& ref_block_idx) {
        unsigned int nblocks_accum = 0;

        for (unsigned int i = 0; i < n_bins; ++i) {
            unsigned int nblocks_per_bin = jag_vec[i].size() / blockDim.x + 1;
            nblocks_accum += nblocks_per_bin;

            if (blockIdx.x < nblocks_accum) {
                bin_idx = i;

                break;
            }

            ref_block_idx += nblocks_per_bin;
        }
    }

    template <typename header_t, typename item_t>
    static __device__ void get_bin_idx(
        const unsigned int& n_bins,
        const device_container<header_t, item_t>& container,
        unsigned int& bin_idx, unsigned int& ref_block_idx) {
        unsigned int nblocks_accum = 0;
        unsigned int nblocks_per_bin = 0;
        for (unsigned int i = 0; i < n_bins; ++i) {
            nblocks_per_bin = container.headers[i] / blockDim.x + 1;
            nblocks_accum += nblocks_per_bin;

            if (blockIdx.x < nblocks_accum) {
                bin_idx = i;

                break;
            }

            ref_block_idx += nblocks_per_bin;
        }
    }
};

}  // namespace cuda
}  // namespace traccc
