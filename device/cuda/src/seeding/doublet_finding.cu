/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/seeding/doublet_finding.cuh>
#include <cuda/utils/cuda_helper.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

__global__ void doublet_finding_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_data,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view);

void doublet_finding(const seedfinder_config& config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_view = get_data(doublet_counter_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
        num_blocks += doublet_counter_container.headers[i] / num_threads + 1;
    }

    unsigned int sh_mem = sizeof(int) * num_threads * 2;

    doublet_finding_kernel<<<num_blocks, num_threads, sh_mem>>>(
        config, internal_sp_view, doublet_counter_view, mid_bot_doublet_view,
        mid_top_doublet_view);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void doublet_finding_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view) {
    device_internal_spacepoint_container internal_sp_device(
        {internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});

    device_doublet_container mid_bot_doublet_device(
        {mid_bot_doublet_view.headers, mid_bot_doublet_view.items});
    device_doublet_container mid_top_doublet_device(
        {mid_top_doublet_view.headers, mid_top_doublet_view.items});

    unsigned int n_bins = internal_sp_device.headers.size();
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;

    cuda_helper::get_bin_idx(n_bins, doublet_counter_device, bin_idx,
                             ref_block_idx);

    const auto& bin_info = internal_sp_device.headers.at(bin_idx);
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);

    auto& num_compat_spM_per_bin = doublet_counter_device.headers.at(bin_idx);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(bin_idx);

    auto& num_mid_bot_doublets_per_bin =
        mid_bot_doublet_device.headers.at(bin_idx);
    auto mid_bot_doublets_per_bin = mid_bot_doublet_device.items.at(bin_idx);

    auto& num_mid_top_doublets_per_bin =
        mid_top_doublet_device.headers.at(bin_idx);
    auto mid_top_doublets_per_bin = mid_top_doublet_device.items.at(bin_idx);

    // zero initialization
    extern __shared__ int num_doublets_per_thread[];
    int* num_mid_bot_doublets_per_thread = num_doublets_per_thread;
    int* num_mid_top_doublets_per_thread =
        &num_mid_bot_doublets_per_thread[blockDim.x];
    num_mid_bot_doublets_per_thread[threadIdx.x] = 0;
    num_mid_top_doublets_per_thread[threadIdx.x] = 0;

    __syncthreads();

    auto gid = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;

    if (gid >= num_compat_spM_per_bin) {
        return;
    }

    auto sp_idx = doublet_counter_per_bin[gid].spM.sp_idx;

    if (sp_idx >= doublet_counter_per_bin.size()) {
        return;
    }

    auto spM_loc = sp_location({bin_idx, sp_idx});
    auto& isp = internal_sp_per_bin[sp_idx];

    unsigned int n_mid_bot_per_spM = 0;
    unsigned int n_mid_top_per_spM = 0;

    unsigned int mid_bot_start_idx = 0;
    unsigned int mid_top_start_idx = 0;

    for (size_t i = 0; i < gid; i++) {
        mid_bot_start_idx += doublet_counter_per_bin[i].n_mid_bot;
        mid_top_start_idx += doublet_counter_per_bin[i].n_mid_top;
    }

    for (size_t i_n = 0; i_n < bin_info.bottom_idx.counts; ++i_n) {
        const auto& neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
        const auto& neigh_internal_sp_per_bin =
            internal_sp_device.items.at(neigh_bin);

        for (size_t spB_idx = 0; spB_idx < neigh_internal_sp_per_bin.size();
             ++spB_idx) {
            const auto& neigh_isp = neigh_internal_sp_per_bin[spB_idx];
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     true)) {
                auto spB_loc = sp_location({neigh_bin, spB_idx});

                if (n_mid_bot_per_spM <
                        doublet_counter_per_bin[gid].n_mid_bot &&
                    num_mid_bot_doublets_per_bin <
                        mid_bot_doublets_per_bin.size()) {
                    size_t pos = mid_bot_start_idx + n_mid_bot_per_spM;
                    if (pos >= mid_bot_doublets_per_bin.size()) {
                        continue;
                    }

                    mid_bot_doublets_per_bin[pos] = doublet({spM_loc, spB_loc});

                    num_mid_bot_doublets_per_thread[threadIdx.x]++;
                    n_mid_bot_per_spM++;
                }
            }

            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     false)) {
                auto spT_loc = sp_location({neigh_bin, spB_idx});

                if (n_mid_top_per_spM <
                        doublet_counter_per_bin[gid].n_mid_top &&
                    num_mid_top_doublets_per_bin <
                        mid_top_doublets_per_bin.size()) {
                    size_t pos = mid_top_start_idx + n_mid_top_per_spM;
                    if (pos >= mid_top_doublets_per_bin.size()) {
                        continue;
                    }

                    mid_top_doublets_per_bin[pos] = doublet({spM_loc, spT_loc});

                    num_mid_top_doublets_per_thread[threadIdx.x]++;
                    n_mid_top_per_spM++;
                }
            }
        }
    }

    __syncthreads();
    cuda_helper::reduce_sum<int>(blockDim.x, threadIdx.x,
                                 num_mid_bot_doublets_per_thread);
    __syncthreads();
    cuda_helper::reduce_sum<int>(blockDim.x, threadIdx.x,
                                 num_mid_top_doublets_per_thread);

    if (threadIdx.x == 0) {
        atomicAdd(&num_mid_bot_doublets_per_bin,
                  num_mid_bot_doublets_per_thread[0]);
        atomicAdd(&num_mid_top_doublets_per_bin,
                  num_mid_top_doublets_per_thread[0]);
    }
}

}  // namespace cuda
}  // namespace traccc
