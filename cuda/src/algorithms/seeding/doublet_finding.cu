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
			    doublet_container_view doublet_data,
			    bool bottom);    
    
void doublet_finding(const seedfinder_config& config,
		     host_internal_spacepoint_container& internal_sp_container,
		     host_doublet_container& doublet_container,
		     bool bottom,
		     vecmem::memory_resource* resource){
    auto internal_sp_data = get_data(internal_sp_container, resource);
    internal_spacepoint_container_view internal_sp_view(internal_sp_data);

    auto doublet_data = get_data(doublet_container, resource);
    doublet_container_view doublet_view(doublet_data);
    
    unsigned int num_threads = WARP_SIZE*2; 
    unsigned int num_blocks = internal_sp_data.headers.m_size;

    doublet_finding_kernel<<< num_blocks, num_threads >>>(config,
							  internal_sp_view,
							  doublet_view,
							  bottom);
       
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());	    
    
}
    
__global__
void doublet_finding_kernel(const seedfinder_config config,
			    internal_spacepoint_container_view internal_sp_data,
			    doublet_container_view doublet_data,
			    bool bottom){
    /*
    // global ID (gid) is for each module
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid>=cell_data.cells.m_size) return;
    
    device_cell_container cells_device({cell_data.modules, cell_data.cells});
    detail::device_label_container labels_device({label_data.counts, label_data.labels});
    auto counts = labels_device.counts;
    auto cells_per_module = cells_device.cells.at(gid);
    auto labels_per_module = labels_device.labels.at(gid);
    
    // run the sparse ccl per module
    counts[gid] = sparse_ccl(cells_per_module, labels_per_module);
    return;	
    */
}
    
}// namespace cuda
}// namespace traccc
