/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/doublet_finding.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc{    
namespace cuda{

__global__
void doublet_finding_kernel(const seedfinder_config config,
			    internal_spacepoint_container_view internal_sp_data,
			    doublet_container_view mid_bot_doublet_view,
			    doublet_container_view mid_top_doublet_view);    
    
void doublet_finding(const seedfinder_config& config,
		     host_internal_spacepoint_container& internal_sp_container,
		     host_doublet_container& mid_bot_doublet_container,
		     host_doublet_container& mid_top_doublet_container,
		     vecmem::memory_resource* resource){
    auto internal_sp_data = get_data(internal_sp_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);
    
    unsigned int num_threads = WARP_SIZE*2; 
    unsigned int num_blocks = internal_sp_data.headers.m_size;

    
    doublet_finding_kernel<<< num_blocks, num_threads >>>(config,
							  internal_sp_data,
							  mid_bot_doublet_view,
							  mid_top_doublet_view);   
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());	        
}
    
__global__
void doublet_finding_kernel(const seedfinder_config config,
			    internal_spacepoint_container_view internal_sp_view,
			    doublet_container_view mid_bot_doublet_view,
			    doublet_container_view mid_top_doublet_view){

    device_internal_spacepoint_container internal_sp_device({internal_sp_view.headers, internal_sp_view.items});
    device_doublet_container mid_bot_doublet_device({mid_bot_doublet_view.headers, mid_bot_doublet_view.items});
    device_doublet_container mid_top_doublet_device({mid_top_doublet_view.headers, mid_top_doublet_view.items});
    
    size_t cur_bin = blockIdx.x;
    
    auto bin_info = internal_sp_device.headers.at(cur_bin);
    auto internal_sp_per_bin = internal_sp_device.items.at(blockIdx.x);

    auto& num_mid_bot_doublets_per_bin = mid_bot_doublet_device.headers.at(blockIdx.x);
    auto mid_bot_doublets_per_bin = mid_bot_doublet_device.items.at(blockIdx.x);

    auto& num_mid_top_doublets_per_bin = mid_top_doublet_device.headers.at(blockIdx.x);
    auto mid_top_doublets_per_bin = mid_top_doublet_device.items.at(blockIdx.x);
    
    size_t n_iter = internal_sp_per_bin.size()/blockDim.x + 1;
    
    for (size_t i_it = 0; i_it < n_iter; ++i_it){

	auto sp_idx = i_it*blockDim.x + threadIdx.x;
	
	if (sp_idx >= internal_sp_per_bin.size()) {
	    continue;
	}

	auto spM_loc = sp_location({blockIdx.x, sp_idx});
	auto isp = internal_sp_per_bin[sp_idx];	

	bool hasCompatBottom = false;
	bool hasCompatTop = false;
     
	for(size_t i_n=0; i_n<bin_info.bottom_idx.counts; ++i_n){	   

	    if (hasCompatBottom && hasCompatTop){
		break;
	    }
	    
	    auto neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
	    
	    auto neigh_internal_sp_per_bin = internal_sp_device.items.at(neigh_bin);		
	    for (size_t spB_idx=0; spB_idx<neigh_internal_sp_per_bin.size(); ++spB_idx){		
		auto neigh_isp = neigh_internal_sp_per_bin[spB_idx];
		if (!hasCompatBottom){
		    hasCompatBottom = doublet_finding_helper::isCompatible(isp, neigh_isp, config, true);
		}
		if (!hasCompatTop){
		    hasCompatTop = doublet_finding_helper::isCompatible(isp, neigh_isp, config, false);
		}
	    }
	}

	// SKIP if there is not compatible hits on any of bottom and top side
	if (!hasCompatBottom || !hasCompatTop) {
	    continue;
	}
	
	for(size_t i_n=0; i_n<bin_info.bottom_idx.counts; ++i_n){		
	    auto neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
	    
	    auto neigh_internal_sp_per_bin = internal_sp_device.items.at(neigh_bin);		
	    for (size_t spB_idx=0; spB_idx<neigh_internal_sp_per_bin.size(); ++spB_idx){	       		
		auto neigh_isp = neigh_internal_sp_per_bin[spB_idx];		

		if (!doublet_finding_helper::isCompatible(isp, neigh_isp, config, true)) continue;
		
		auto spB_loc = sp_location({neigh_bin, spB_idx});
		auto lin = doublet_finding_helper::transform_coordinates(isp, neigh_isp, true);
		auto pos = atomicAdd(&num_mid_bot_doublets_per_bin,1);
	       
		if (pos>=mid_bot_doublets_per_bin.size()){
		    num_mid_bot_doublets_per_bin = mid_bot_doublets_per_bin.size();
		    continue;
		}
		
		mid_bot_doublets_per_bin[pos] = doublet({spM_loc, spB_loc, lin});
		
		
	    }				
	}

	for(size_t i_n=0; i_n<bin_info.top_idx.counts; ++i_n){		
	    auto neigh_bin = bin_info.top_idx.vector_indices[i_n];
	    
	    auto neigh_internal_sp_per_bin = internal_sp_device.items.at(neigh_bin);		
	    for (size_t spT_idx=0; spT_idx<neigh_internal_sp_per_bin.size(); ++spT_idx){		
		auto neigh_isp = neigh_internal_sp_per_bin[spT_idx];		
	    
		if (!doublet_finding_helper::isCompatible(isp, neigh_isp, config, false)) continue;		
		auto lin = doublet_finding_helper::transform_coordinates(isp, neigh_isp, false);
		auto spT_loc = sp_location({neigh_bin, spT_idx});

		auto pos = atomicAdd(&num_mid_top_doublets_per_bin,1);

		if (pos>=mid_top_doublets_per_bin.size()){
		    num_mid_top_doublets_per_bin = mid_top_doublets_per_bin.size();
		    continue;
		}
		
		mid_top_doublets_per_bin[pos] = doublet({spM_loc, spT_loc, lin});   
	    }				
	}	   
    }

    
    //bubble sort in terms of spM_idx
    //To-do: make a function for bubble sort
    //__syncthreads();
    //int tid = threadIdx.x;
    //int n_sort_iter;
    //doublet tempVal;
    /*
    n_sort_iter = num_mid_bot_doublets_per_bin/blockDim.x + 1;
    //n_sort_iter = 1;
    for (int i = 0; i < num_mid_bot_doublets_per_bin / 2 + 1; i++) {
	if (threadIdx.x < num_mid_bot_doublets_per_bin) {
	    for (int j=0; j<n_sort_iter; j++){
		int k = j*blockDim.x + tid;		
		if (k % 2 == 0 && k < num_mid_bot_doublets_per_bin - 1) {
		    if (mid_bot_doublets_per_bin[k + 1].sp1.sp_idx < mid_bot_doublets_per_bin[k].sp1.sp_idx) {
			tempVal = mid_bot_doublets_per_bin[k];
			mid_bot_doublets_per_bin[k] = mid_bot_doublets_per_bin[k + 1];
			mid_bot_doublets_per_bin[k + 1] = tempVal;
		    }
		}
	    }
	}
	__syncthreads();
	if (threadIdx.x < num_mid_bot_doublets_per_bin) {
	    for (int j=0; j<n_sort_iter; j++){
		int k = j*blockDim.x + tid;				
		if (k % 2 == 1 && k < num_mid_bot_doublets_per_bin - 1) {	 
		    if (mid_bot_doublets_per_bin[k + 1].sp1.sp_idx < mid_bot_doublets_per_bin[k].sp1.sp_idx) {
			tempVal = mid_bot_doublets_per_bin[k];
			mid_bot_doublets_per_bin[k] = mid_bot_doublets_per_bin[k + 1];
			mid_bot_doublets_per_bin[k + 1] = tempVal;
		    }
		}
	    }
	}
	__syncthreads();
    }
    __syncthreads();

    n_sort_iter = num_mid_top_doublets_per_bin/blockDim.x + 1;
    //n_sort_iter = 1;
    for (int i = 0; i < num_mid_top_doublets_per_bin / 2 + 1; i++) {
	if (threadIdx.x < num_mid_top_doublets_per_bin) {
	    for (int j=0; j<n_sort_iter; j++){
		int k = j*blockDim.x + tid;		
		if (k % 2 == 0 && k < num_mid_top_doublets_per_bin - 1) {
		    if (mid_top_doublets_per_bin[k + 1].sp1.sp_idx < mid_top_doublets_per_bin[k].sp1.sp_idx) {
			tempVal = mid_top_doublets_per_bin[k];
			mid_top_doublets_per_bin[k] = mid_top_doublets_per_bin[k + 1];
			mid_top_doublets_per_bin[k + 1] = tempVal;
		    }
		}
	    }
	}
	__syncthreads();
	if (threadIdx.x < num_mid_top_doublets_per_bin) {
	    for (int j=0; j<n_sort_iter; j++){
		int k = j*blockDim.x + tid;				
		if (k % 2 == 1 && k < num_mid_top_doublets_per_bin - 1) {	 
		    if (mid_top_doublets_per_bin[k + 1].sp1.sp_idx < mid_top_doublets_per_bin[k].sp1.sp_idx) {
			tempVal = mid_top_doublets_per_bin[k];
			mid_top_doublets_per_bin[k] = mid_top_doublets_per_bin[k + 1];
			mid_top_doublets_per_bin[k + 1] = tempVal;
		    }
		}
	    }
	}
	__syncthreads();
    }
    __syncthreads();
    */    
    
    /*
    if (threadIdx.x == 0 && blockIdx.x==76){
	printf("%d \n", num_mid_bot_doublets_per_bin);
	for (auto el: mid_bot_doublets_per_bin){
	    printf("%d ", el.sp1.sp_idx);
	}
	printf("\n");
    }
    */
    /*
    if (threadIdx.x == 0 && blockIdx.x==76){
	for (auto el: mid_top_doublets_per_bin){
	    printf("%d ", el.sp1.sp_idx);
	}
	printf("\n");
    }
    */
}
    
}// namespace cuda
}// namespace traccc
