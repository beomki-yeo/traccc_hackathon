/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/doublet_counting.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc{    
namespace cuda{

__global__
void doublet_counting_kernel(const seedfinder_config config,
			     internal_spacepoint_container_view internal_sp_view,
			     doublet_counter_container_view doublet_count_view);
    
void doublet_counting(const seedfinder_config& config,
		      host_internal_spacepoint_container& internal_sp_container,
		      host_doublet_counter_container& doublet_counter_container,
		      vecmem::memory_resource* resource){

    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view = get_data(doublet_counter_container, resource);
    
    unsigned int num_threads = WARP_SIZE*4; 
    unsigned int num_blocks = internal_sp_view.headers.m_size;
    
    doublet_counting_kernel<<< num_blocks, num_threads >>>(config,
							   internal_sp_view,
							   doublet_counter_container_view);
    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());	        
    
}

__global__
void doublet_counting_kernel(const seedfinder_config config,
			     internal_spacepoint_container_view internal_sp_view,
			     doublet_counter_container_view doublet_counter_view){

    device_internal_spacepoint_container internal_sp_device({internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device({doublet_counter_view.headers, doublet_counter_view.items});

    auto bin_info = internal_sp_device.headers.at(blockIdx.x);
    auto internal_sp_per_bin = internal_sp_device.items.at(blockIdx.x);
    auto& num_compat_spM_per_bin = doublet_counter_device.headers.at(blockIdx.x);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(blockIdx.x);

    num_compat_spM_per_bin = 0;

    size_t n_iter = internal_sp_per_bin.size()/blockDim.x + 1;

    //size_t n_iter = doublet_counter_per_bin.size()/blockDim.x + 1;

    __syncthreads();
    
    for (size_t i_it = 0; i_it < n_iter; ++i_it){
	auto sp_idx = i_it*blockDim.x + threadIdx.x;

	if (sp_idx >= internal_sp_per_bin.size()) {
	    continue;
	}
	
	/*
	if (sp_idx >= doublet_counter_per_bin.size()) {
	    continue;
	}
	*/
		
	auto spM_loc = sp_location({blockIdx.x, sp_idx});
	auto isp = internal_sp_per_bin[sp_idx];	

	doublet_counter_per_bin[sp_idx].spM = spM_loc;
	doublet_counter_per_bin[sp_idx].n_mid_bot = 0;
	doublet_counter_per_bin[sp_idx].n_mid_top = 0;
	
	for(size_t i_n=0; i_n<bin_info.bottom_idx.counts; ++i_n){

	    auto neigh_bin = bin_info.bottom_idx.vector_indices[i_n];	    
	    auto neigh_internal_sp_per_bin = internal_sp_device.items.at(neigh_bin);
	    
	    for (size_t spB_idx=0; spB_idx<neigh_internal_sp_per_bin.size(); ++spB_idx){
		auto neigh_isp = neigh_internal_sp_per_bin[spB_idx];
		if (doublet_finding_helper::isCompatible(isp, neigh_isp, config, true)){
		    doublet_counter_per_bin[sp_idx].n_mid_bot++;
		}
		
		if (doublet_finding_helper::isCompatible(isp, neigh_isp, config, false)){
		    doublet_counter_per_bin[sp_idx].n_mid_top++;
		}

	    }
	}
	
	if (doublet_counter_per_bin[sp_idx].n_mid_bot > 0 &&
	    doublet_counter_per_bin[sp_idx].n_mid_top > 0){
	    atomicAdd(&num_compat_spM_per_bin,1);
	}
    }    
}
    
}// namespace cuda
}// namespace traccc
