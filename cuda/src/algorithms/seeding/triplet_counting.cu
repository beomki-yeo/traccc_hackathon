/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/triplet_counting.cuh>
#include <cuda/utils/cuda_helper.cuh>

namespace traccc {
namespace cuda {

__global__ void triplet_counting_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view,
    triplet_counter_container_view triplet_counter_view);

void triplet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      host_doublet_container& mid_bot_doublet_container,
                      host_doublet_container& mid_top_doublet_container,
                      host_triplet_counter_container& triplet_counter_container,
                      vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);
    auto triplet_counter_container_view =
        get_data(triplet_counter_container, resource);

    unsigned int num_threads = WARP_SIZE * 8;
    unsigned int num_blocks = 0;
    for (size_t i=0; i<internal_sp_view.headers.m_size; ++i){
	num_blocks += mid_bot_doublet_container.headers[i] / num_threads +1;
    }
    
    triplet_counting_kernel<<<num_blocks, num_threads>>>(
        config, internal_sp_view, doublet_counter_container_view,
        mid_bot_doublet_view, mid_top_doublet_view,
        triplet_counter_container_view);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void triplet_counting_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view,
    triplet_counter_container_view triplet_counter_view) {
    device_internal_spacepoint_container internal_sp_device(
        {internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});
    device_doublet_container mid_bot_doublet_device(
        {mid_bot_doublet_view.headers, mid_bot_doublet_view.items});
    device_doublet_container mid_top_doublet_device(
        {mid_top_doublet_view.headers, mid_top_doublet_view.items});
    device_triplet_counter_container triplet_counter_device(
        {triplet_counter_view.headers, triplet_counter_view.items});

    
    unsigned int n_bins = internal_sp_device.headers.size();
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;

    cuda_helper::get_bin_idx(n_bins,
			     mid_bot_doublet_device,
			     bin_idx,
			     ref_block_idx);
    
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);
    auto& num_compat_spM_per_bin =
        doublet_counter_device.headers.at(bin_idx);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(bin_idx);
    const auto& num_mid_bot_doublets_per_bin =
        mid_bot_doublet_device.headers.at(bin_idx);
    auto mid_bot_doublets_per_bin = mid_bot_doublet_device.items.at(bin_idx);
    const auto& num_mid_top_doublets_per_bin =
        mid_top_doublet_device.headers.at(bin_idx);
    auto mid_top_doublets_per_bin = mid_top_doublet_device.items.at(bin_idx);
    auto& num_compat_mb_per_bin = triplet_counter_device.headers.at(bin_idx);
    auto triplet_counter_per_bin = triplet_counter_device.items.at(bin_idx);    
    
    __syncthreads();

    auto mb_idx = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;
    
    if (mb_idx >= num_mid_bot_doublets_per_bin) {
	return;
    }
    
    auto& mid_bot_doublet = mid_bot_doublets_per_bin[mb_idx];
    
    auto& spM_idx = mid_bot_doublet.sp1.sp_idx;
    auto& spM = internal_sp_per_bin[spM_idx];
    
    auto& spB_bin = mid_bot_doublet.sp2.bin_idx;
    auto& spB_idx = mid_bot_doublet.sp2.sp_idx;
    auto& spB = internal_sp_device.items.at(spB_bin)[spB_idx];
    
    auto lb = doublet_finding_helper::transform_coordinates(spM, spB, true);
    
    scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
    scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2;
    scatteringInRegion2 *= config.sigmaScattering * config.sigmaScattering;
    scalar curvature, impact_parameter;
    
    unsigned int mb_end_idx = 0;
    unsigned int mt_start_idx = 0;
    unsigned int mt_end_idx = 0;
    
    for (unsigned int i = 0; i < num_compat_spM_per_bin; ++i) {
	mb_end_idx += doublet_counter_per_bin[i].n_mid_bot;
	mt_end_idx += doublet_counter_per_bin[i].n_mid_top;
	
	if (mb_end_idx > mb_idx) {
	    break;
	}
	mt_start_idx += doublet_counter_per_bin[i].n_mid_top;
    }
    
    if (mt_end_idx >= mid_top_doublets_per_bin.size()) {
	mt_end_idx = fmin(mid_top_doublets_per_bin.size(), mt_end_idx);
    }
    
    if (mt_start_idx >= mid_top_doublets_per_bin.size()) {
	return;
    }
    
    unsigned int n_triplets = 0;
    
    // iterate over mid-top doublets
    for (unsigned int i = mt_start_idx; i < mt_end_idx; ++i) {
	auto& mid_top_doublet = mid_top_doublets_per_bin[i];
	
	auto& spT_bin = mid_top_doublet.sp2.bin_idx;
	auto& spT_idx = mid_top_doublet.sp2.sp_idx;
	auto& spT = internal_sp_device.items.at(spT_bin)[spT_idx];
	
	auto lt =
	    doublet_finding_helper::transform_coordinates(spM, spT, false);
	
	if (triplet_finding_helper::isCompatible(
						 spM, lb, lt, config, iSinTheta2, scatteringInRegion2,
						 curvature, impact_parameter)) {
	    n_triplets++;
	}
    }
    
    if (n_triplets > 0) {
	auto pos = atomicAdd(&num_compat_mb_per_bin, 1);
	triplet_counter_per_bin[pos].n_triplets = n_triplets;
	triplet_counter_per_bin[pos].mid_bot_doublet = mid_bot_doublet;
    }
}

}  // namespace cuda
}  // namespace traccc
