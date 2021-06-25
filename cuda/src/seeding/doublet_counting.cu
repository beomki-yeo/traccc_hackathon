/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/seeding/doublet_counting.cuh>
#include <cuda/utils/cuda_helper.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

__global__ void doublet_counting_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_count_view);

void doublet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);

    unsigned int num_threads = WARP_SIZE * 2;
    
    unsigned int num_blocks = 0;
    for (size_t i=0; i<internal_sp_view.headers.size(); ++i){
	num_blocks += internal_sp_view.items.m_ptr[i].size() / num_threads + 1;
    }
        
    doublet_counting_kernel<<<num_blocks, num_threads>>>(
        config, internal_sp_view, doublet_counter_container_view);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void doublet_counting_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view) {
    
    device_internal_spacepoint_container internal_sp_device(
        {internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});

    unsigned int n_bins = internal_sp_device.headers.size();
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;

    cuda_helper::get_bin_idx(n_bins, internal_sp_device.items, bin_idx, ref_block_idx);
    
    const auto& bin_info = internal_sp_device.headers.at(bin_idx);
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);
    auto& num_compat_spM_per_bin =
        doublet_counter_device.headers.at(bin_idx);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(bin_idx);

    // zero initialization
    __syncthreads();

    auto sp_idx = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;

    if (sp_idx >= doublet_counter_per_bin.size()) {
	return;
    }
    
    unsigned int n_mid_bot = 0;
    unsigned int n_mid_top = 0;
    
    auto spM_loc = sp_location({bin_idx, sp_idx});
    const auto& isp = internal_sp_per_bin[sp_idx];
    
    doublet_counter_per_bin[sp_idx].n_mid_bot = 0;
    doublet_counter_per_bin[sp_idx].n_mid_top = 0;
    
    for (size_t i_n = 0; i_n < bin_info.bottom_idx.counts; ++i_n) {
	const auto& neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
	const auto& neigh_internal_sp_per_bin =
	    internal_sp_device.items.at(neigh_bin);
	
	for (size_t spB_idx = 0; spB_idx < neigh_internal_sp_per_bin.size();
	     ++spB_idx) {
	    const auto& neigh_isp = neigh_internal_sp_per_bin[spB_idx];
	    if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
						     true)) {
		n_mid_bot++;
	    }
	    
	    if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
						     false)) {
		n_mid_top++;
	    }
	}
    }
    
    if (n_mid_bot > 0 && n_mid_top > 0) {
	auto pos = atomicAdd(&num_compat_spM_per_bin, 1);	
	doublet_counter_per_bin[pos].spM = spM_loc;
	doublet_counter_per_bin[pos].n_mid_bot = n_mid_bot;
	doublet_counter_per_bin[pos].n_mid_top = n_mid_top;
    }
}

}  // namespace cuda
}  // namespace traccc
