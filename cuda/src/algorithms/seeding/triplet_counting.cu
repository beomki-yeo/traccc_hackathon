/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/triplet_counting.cuh>
#include <cuda/utils/cuda_helper.cuh>

namespace traccc{    
namespace cuda{

__global__
void triplet_counting_kernel(const seedfinder_config config,
			     const seedfilter_config filter_config,
			     internal_spacepoint_container_view internal_sp_view,
			     doublet_counter_container_view doublet_counter_view,
			     doublet_container_view mid_bot_doublet_view,
			     doublet_container_view mid_top_doublet_view,
			     triplet_counter_container_view triplet_counter_view);
    
void triplet_counting(const seedfinder_config& config,
		      const seedfilter_config& filter_config,
		      host_internal_spacepoint_container& internal_sp_container,
		      host_doublet_counter_container& doublet_counter_container,
		      host_doublet_container& mid_bot_doublet_container,
		      host_doublet_container& mid_top_doublet_container,
		      host_triplet_counter_container& triplet_counter_container,
		      vecmem::memory_resource* resource){
    
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view = get_data(doublet_counter_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);
    auto triplet_counter_container_view = get_data(triplet_counter_container, resource);
    
    unsigned int num_threads = WARP_SIZE*8; 
    unsigned int num_blocks = internal_sp_view.headers.m_size;
    
    triplet_counting_kernel
	<<< num_blocks,num_threads >>>(config,
				       filter_config,
				       internal_sp_view,
				       doublet_counter_container_view,
				       mid_bot_doublet_view,
				       mid_top_doublet_view,
				       triplet_counter_container_view);
    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());	            
}

__global__
void triplet_counting_kernel(const seedfinder_config config,
			     const seedfilter_config filter_config,
			     internal_spacepoint_container_view internal_sp_view,
			     doublet_counter_container_view doublet_counter_view,
			     doublet_container_view mid_bot_doublet_view,
			     doublet_container_view mid_top_doublet_view,
			     triplet_counter_container_view triplet_counter_view){

    device_internal_spacepoint_container internal_sp_device({internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device({doublet_counter_view.headers, doublet_counter_view.items});	
    device_doublet_container mid_bot_doublet_device({mid_bot_doublet_view.headers, mid_bot_doublet_view.items});
    device_doublet_container mid_top_doublet_device({mid_top_doublet_view.headers, mid_top_doublet_view.items});
    device_triplet_counter_container triplet_counter_device({triplet_counter_view.headers, triplet_counter_view.items});
    
    auto bin_info = internal_sp_device.headers.at(blockIdx.x);
    auto internal_sp_per_bin = internal_sp_device.items.at(blockIdx.x);
    auto& num_compat_spM_per_bin = doublet_counter_device.headers.at(blockIdx.x);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(blockIdx.x);
    auto num_mid_bot_doublets_per_bin = mid_bot_doublet_device.headers.at(blockIdx.x);
    auto mid_bot_doublets_per_bin = mid_bot_doublet_device.items.at(blockIdx.x);
    auto num_mid_top_doublets_per_bin = mid_top_doublet_device.headers.at(blockIdx.x);
    auto mid_top_doublets_per_bin = mid_top_doublet_device.items.at(blockIdx.x);
    auto& num_compat_mb_per_bin = triplet_counter_device.headers.at(blockIdx.x);
    auto triplet_counter_per_bin = triplet_counter_device.items.at(blockIdx.x);    
    
    size_t n_iter = num_mid_bot_doublets_per_bin/blockDim.x + 1;

    // zero initialization
    num_compat_mb_per_bin = 0;
    
    __syncthreads();
    
    for (size_t i_it = 0; i_it < n_iter; ++i_it){
	auto mb_idx = i_it*blockDim.x + threadIdx.x;
	auto& mid_bot_doublet = mid_bot_doublets_per_bin[mb_idx];
	
	if (mb_idx >= num_mid_bot_doublets_per_bin){
	    continue;
	}

	size_t num_triplets_per_mid_bot = 0;
	auto& spM_idx = mid_bot_doublet.sp1.sp_idx;
	auto& spM = internal_sp_per_bin[spM_idx];
	auto& lb = mid_bot_doublet.lin;
	
	scalar iSinTheta2 = 1 + lb.cotTheta * lb.cotTheta;
	scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2;
	scatteringInRegion2 *= config.sigmaScattering * config.sigmaScattering;
	scalar curvature, impact_parameter;	
	
	size_t mb_end_idx = 0;
	size_t mt_start_idx = 0;
	size_t mt_end_idx = 0;

	for (int i=0; i<num_compat_spM_per_bin; ++i){

	    mb_end_idx += doublet_counter_per_bin[i].n_mid_bot;
	    mt_end_idx += doublet_counter_per_bin[i].n_mid_top;
	    
	    if (mb_end_idx > mb_idx){
		break;
	    }
	    mt_start_idx += doublet_counter_per_bin[i].n_mid_top;
	}
	
	if (mt_end_idx >= mid_top_doublets_per_bin.size()){
	    mt_end_idx = fmin(mid_top_doublets_per_bin.size(), mt_end_idx);
	}	    
	
	if (mt_start_idx >= mid_top_doublets_per_bin.size()){
	    continue;
	}

	int n_triplets = 0;
	
	// iterate over mid-top doublets	
	for (auto mt_it = mid_top_doublets_per_bin.begin()+mt_start_idx;
	     mt_it!= mid_top_doublets_per_bin.begin()+mt_end_idx ;
	     mt_it++){
	    
	    auto lt = (*mt_it).lin;
	    
	    if (triplet_finding_helper::isCompatible(spM, lb, lt, config,
						      iSinTheta2, scatteringInRegion2,
						      curvature, impact_parameter)){
		n_triplets++;
	    }	    
	}

	if (n_triplets > 0){
	    auto pos = atomicAdd(&num_compat_mb_per_bin,1);
	    triplet_counter_per_bin[pos].n_triplets = n_triplets;
	    triplet_counter_per_bin[pos].mid_bot_doublet = mid_bot_doublet;
	}	
    }
}
    
}// namespace cuda
}// namespace traccc