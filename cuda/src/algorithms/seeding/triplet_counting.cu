/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/triplet_counting.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc{    
namespace cuda{

__global__
void triplet_counting_kernel(const seedfinder_config config,
			     const seedfilter_config filter_config,
			     internal_spacepoint_container_view internal_sp_view,
			     doublet_container_view mid_bot_doublet_view,
			     doublet_container_view mid_top_doublet_view,
			     triplet_counter_container_view triplet_counter_view);
    
void triplet_counting(const seedfinder_config& config,
		      const seedfilter_config& filter_config,
		      host_internal_spacepoint_container& internal_sp_container,
		      host_doublet_container& mid_bot_doublet_container,
		      host_doublet_container& mid_top_doublet_container,
		      host_triplet_counter_container& triplet_counter_container,
		      vecmem::memory_resource* resource){
    
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);
    auto triplet_counter_container_view = get_data(triplet_counter_container, resource);
    
    unsigned int num_threads = WARP_SIZE*4; 
    unsigned int num_blocks = internal_sp_view.headers.m_size;
    
    triplet_counting_kernel<<< num_blocks, num_threads >>>(config,
							   filter_config,
							   internal_sp_view,
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
			     doublet_container_view mid_bot_doublet_view,
			     doublet_container_view mid_top_doublet_view,
			     triplet_counter_container_view triplet_counter_view){

    

}
    
}// namespace cuda
}// namespace traccc
