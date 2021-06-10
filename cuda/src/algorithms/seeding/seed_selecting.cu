/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/seed_selecting.cuh>
#include <cuda/utils/cuda_helper.cuh>

namespace traccc{    
namespace cuda{

__global__
void seed_selecting_kernel(const seedfilter_config filter_config,
			   internal_spacepoint_container_view internal_sp_view,
			   doublet_counter_container_view doublet_counter_view,
			   triplet_counter_container_view triplet_counter_view,
			   triplet_container_view triplet_view,
			   seed_container_view seed_view
			   );

    
void seed_selecting(const seedfilter_config& filter_config,
		    host_internal_spacepoint_container& internal_sp_container,
		    host_doublet_counter_container& doublet_counter_container,
		    host_triplet_counter_container& triplet_counter_container,
		    host_triplet_container& triplet_container,
		    host_seed_container& seed_container,
		    vecmem::memory_resource* resource){

    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view = get_data(doublet_counter_container, resource);
    auto triplet_counter_container_view = get_data(triplet_counter_container, resource);
    auto triplet_container_view = get_data(triplet_container, resource);
    auto seed_container_view = get_data(seed_container, resource);    
    
    unsigned int num_threads = WARP_SIZE*4; 
    unsigned int num_blocks = internal_sp_view.headers.m_size;
    
    seed_selecting_kernel
	<<< num_blocks,num_threads >>>(filter_config,
				       internal_sp_view,
				       doublet_counter_container_view,
				       triplet_counter_container_view,
				       triplet_container_view,
				       seed_container_view);    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());	                    
}

__global__
void seed_selecting_kernel(const seedfilter_config filter_config,
			   internal_spacepoint_container_view internal_sp_view,
			   doublet_counter_container_view doublet_counter_view,
			   triplet_counter_container_view triplet_counter_view,
			   triplet_container_view triplet_view,
			   seed_container_view seed_view
			   ){
    device_internal_spacepoint_container internal_sp_device({internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device({doublet_counter_view.headers, doublet_counter_view.items});	
    device_triplet_counter_container triplet_counter_device({triplet_counter_view.headers, triplet_counter_view.items});
    device_triplet_container triplet_device({triplet_view.headers, triplet_view.items});
    device_seed_container seed_device({seed_view.headers, seed_view.items});

    auto bin_info = internal_sp_device.headers.at(blockIdx.x);
    auto internal_sp_per_bin = internal_sp_device.items.at(blockIdx.x);
    auto& num_compat_spM_per_bin = doublet_counter_device.headers.at(blockIdx.x);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(blockIdx.x);    

    auto& num_compat_mb_per_bin = triplet_counter_device.headers.at(blockIdx.x);
    auto triplet_counter_per_bin = triplet_counter_device.items.at(blockIdx.x);    
    
    auto& num_triplets_per_bin = triplet_device.headers.at(blockIdx.x);    
    auto triplets_per_bin = triplet_device.items.at(blockIdx.x);

    auto& num_seeds = seed_device.headers.at(0);
    auto seeds = seed_device.items.at(0);    
    
}

    
}// namespace cuda
}// namespace traccc
