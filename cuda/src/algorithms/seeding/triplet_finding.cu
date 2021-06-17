/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/triplet_finding.cuh>
#include <cuda/utils/cuda_helper.cuh>

namespace traccc {
namespace cuda {

__global__ void triplet_finding_kernel(
    const seedfinder_config config, const seedfilter_config filter_config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view);

void triplet_finding(const seedfinder_config& config,
                     const seedfilter_config& filter_config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_view = get_data(doublet_counter_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);
    auto triplet_counter_view = get_data(triplet_counter_container, resource);
    auto triplet_view = get_data(triplet_container, resource);

    unsigned int num_threads = WARP_SIZE * 8;
    //unsigned int num_blocks = internal_sp_view.headers.m_size;

    unsigned int num_blocks = 0;
    for (size_t i=0; i<internal_sp_view.headers.m_size; ++i){
	num_blocks += triplet_counter_view.items.m_ptr[i].m_size / num_threads +1;
    }
    
    unsigned int sh_mem = sizeof(int) * num_threads;

    triplet_finding_kernel<<<num_blocks, num_threads, sh_mem>>>(
        config, filter_config, internal_sp_view, doublet_counter_view,
        mid_bot_doublet_view, mid_top_doublet_view, triplet_counter_view,
        triplet_view);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void triplet_finding_kernel(
    const seedfinder_config config, const seedfilter_config filter_config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view) {
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
    device_triplet_container triplet_device(
        {triplet_view.headers, triplet_view.items});

    unsigned int n_bins = internal_sp_device.headers.size();
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;

    cuda_helper::get_bin_idx(n_bins,
			     triplet_counter_device,
			     bin_idx,
			     ref_block_idx);
    
    auto bin_info = internal_sp_device.headers.at(bin_idx);
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);
    auto& num_compat_spM_per_bin =
        doublet_counter_device.headers.at(bin_idx);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(bin_idx);
    auto num_mid_bot_doublets_per_bin =
        mid_bot_doublet_device.headers.at(bin_idx);
    auto mid_bot_doublets_per_bin = mid_bot_doublet_device.items.at(bin_idx);
    auto num_mid_top_doublets_per_bin =
        mid_top_doublet_device.headers.at(bin_idx);
    auto mid_top_doublets_per_bin = mid_top_doublet_device.items.at(bin_idx);

    auto& num_compat_mb_per_bin = triplet_counter_device.headers.at(bin_idx);
    auto triplet_counter_per_bin = triplet_counter_device.items.at(bin_idx);

    auto& num_triplets_per_bin = triplet_device.headers.at(bin_idx);
    auto triplets_per_bin = triplet_device.items.at(bin_idx);

    //size_t n_iter = num_compat_mb_per_bin / blockDim.x + 1;

    extern __shared__ int num_triplets_per_thread[];
    num_triplets_per_thread[threadIdx.x] = 0;

    __syncthreads();

    //for (size_t i_it = 0; i_it < n_iter; ++i_it) {
    //auto gid = i_it * blockDim.x + threadIdx.x;
    auto gid = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;
    if (gid >= num_compat_mb_per_bin) {
	//continue;
	return;
    }
    auto& mid_bot_doublet = triplet_counter_per_bin[gid].mid_bot_doublet;

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
    unsigned int ref_idx;
    
    for (unsigned int i = 0; i < num_mid_bot_doublets_per_bin; i++) {
	if (mid_bot_doublet == mid_bot_doublets_per_bin[i]) {
	    ref_idx = i;
	    break;
	}
    }
    
    for (unsigned int i = 0; i < num_compat_spM_per_bin; ++i) {
	mb_end_idx += doublet_counter_per_bin[i].n_mid_bot;
	mt_end_idx += doublet_counter_per_bin[i].n_mid_top;
	
	if (mb_end_idx > ref_idx) {
	    break;
	}
	
	mt_start_idx += doublet_counter_per_bin[i].n_mid_top;
    }
    
    if (mt_end_idx >= mid_top_doublets_per_bin.size()) {
	mt_end_idx = fmin(mid_top_doublets_per_bin.size(), mt_end_idx);
    }
    
    if (mt_start_idx >= mid_top_doublets_per_bin.size()) {
	//continue;
	return;
    }
    
    unsigned int n_triplets_per_mb = 0;
    unsigned int triplet_start_idx = 0;
    
    for (unsigned int i = 0; i < gid; i++) {
	triplet_start_idx += triplet_counter_per_bin[i].n_triplets;
    }
    
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
	    size_t pos = triplet_start_idx + n_triplets_per_mb;
	    if (pos >= triplets_per_bin.size()) {
		continue;
	    }
	    
	    triplets_per_bin[pos] = triplet(
					    {mid_bot_doublet.sp2, mid_bot_doublet.sp1,
					     mid_top_doublet.sp2, curvature,
					     -impact_parameter * filter_config.impactWeightFactor,
					     lb.Zo()});
	    
	    num_triplets_per_thread[threadIdx.x]++;
	    n_triplets_per_mb++;
	}
    }


    __syncthreads();
    cuda_helper::reduce_sum<int>(blockDim.x, threadIdx.x,
                                 num_triplets_per_thread);

    if (threadIdx.x == 0) {
        //num_triplets_per_bin = num_triplets_per_thread[0];
	atomicAdd(&num_triplets_per_bin, num_triplets_per_thread[0]);
    }
}

}  // namespace cuda
}  // namespace traccc
