/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <cuda/algorithms/seeding/triplet_finding.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc{    
namespace cuda{

__global__
void triplet_finding_kernel(const seedfinder_config config,
			    const seedfilter_config filter_config,
			    internal_spacepoint_container_view internal_sp_view,
			    doublet_container_view mid_bot_doublet_view,
			    doublet_container_view mid_top_doublet_view,
			    vecmem::data::jagged_vector_view< size_t > n_mb_per_spM_view,
			    vecmem::data::jagged_vector_view< size_t > n_mt_per_spM_view,
			    triplet_container_view triplet_view);    

    
void triplet_finding(const seedfinder_config& config,
		     const seedfilter_config& filter_config,
		     host_internal_spacepoint_container& internal_sp_container,
		     host_doublet_container& mid_bot_doublet_container,
		     host_doublet_container& mid_top_doublet_container,
		     vecmem::jagged_vector< size_t >& n_mb_per_spM,  
		     vecmem::jagged_vector< size_t >& n_mt_per_spM,
		     host_triplet_container& triplet_container,
		     vecmem::memory_resource* resource){

    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);
    auto n_mb_per_spM_view = vecmem::get_data(n_mb_per_spM, resource);
    auto n_mt_per_spM_view = vecmem::get_data(n_mt_per_spM, resource);
    
    auto triplet_view = get_data(triplet_container, resource);
    
    unsigned int num_threads = WARP_SIZE*2; 
    unsigned int num_blocks = internal_sp_view.headers.m_size;

    triplet_finding_kernel<<< num_blocks, num_threads >>>(config,
							  filter_config,
							  internal_sp_view,
							  mid_bot_doublet_view,
							  mid_top_doublet_view,
							  n_mb_per_spM_view,
							  n_mt_per_spM_view,
							  triplet_view);
    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());	            
}

__global__
void triplet_finding_kernel(const seedfinder_config config,
			    const seedfilter_config filter_config,
			    internal_spacepoint_container_view internal_sp_view,
			    doublet_container_view mid_bot_doublet_view,
			    doublet_container_view mid_top_doublet_view,	   
			    vecmem::data::jagged_vector_view< size_t > n_mb_per_spM_view,
			    vecmem::data::jagged_vector_view< size_t > n_mt_per_spM_view,
			    triplet_container_view triplet_view){

    device_internal_spacepoint_container internal_sp_device({internal_sp_view.headers, internal_sp_view.items});
    device_doublet_container mid_bot_doublet_device({mid_bot_doublet_view.headers, mid_bot_doublet_view.items});
    device_doublet_container mid_top_doublet_device({mid_top_doublet_view.headers, mid_top_doublet_view.items});
    vecmem::jagged_device_vector< size_t > n_mb_per_spM_device(n_mb_per_spM_view);
    vecmem::jagged_device_vector< size_t > n_mt_per_spM_device(n_mt_per_spM_view);
    
    device_triplet_container triplet_device({triplet_view.headers, triplet_view.items});
    
    auto bin_info = internal_sp_device.headers.at(blockIdx.x);
    auto internal_sp_per_bin = internal_sp_device.items.at(blockIdx.x);
    auto num_mid_bot_doublets_per_bin = mid_bot_doublet_device.headers.at(blockIdx.x);
    auto mid_bot_doublets_per_bin = mid_bot_doublet_device.items.at(blockIdx.x);
    auto num_mid_top_doublets_per_bin = mid_top_doublet_device.headers.at(blockIdx.x);
    auto mid_top_doublets_per_bin = mid_top_doublet_device.items.at(blockIdx.x);
   
    size_t n_iter = num_mid_bot_doublets_per_bin/blockDim.x + 1;

    auto& num_triplets_per_bin = triplet_device.headers.at(blockIdx.x);
    
    auto triplets_per_bin = triplet_device.items.at(blockIdx.x);

    auto n_mb_per_spM = n_mb_per_spM_device.at(blockIdx.x);
    auto n_mt_per_spM = n_mt_per_spM_device.at(blockIdx.x);
    
    for (size_t i_it = 0; i_it < n_iter; ++i_it){
	auto mb_idx = i_it*blockDim.x + threadIdx.x;
	auto mid_bot_doublet = mid_bot_doublets_per_bin[mb_idx];
	
	if (mb_idx >= num_mid_bot_doublets_per_bin){
	    continue;
	}

	if (n_mb_per_spM.size() == 0 || n_mt_per_spM.size() == 0){
	    continue;
	}

	size_t num_triplets_per_mid_bot = 0;
	auto spM = internal_sp_per_bin[mid_bot_doublet.sp1.sp_idx];
	auto lb = mid_bot_doublet.lin;

	scalar iSinTheta2 = 1 + lb.cotTheta * lb.cotTheta;
	scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2;
	scatteringInRegion2 *= config.sigmaScattering * config.sigmaScattering;
	scalar curvature, impact_parameter;	
	

	size_t mb_end_idx = 0;
	size_t mt_start_idx = 0;
	size_t mt_end_idx = 0;	
	
	for (int i=0; i<n_mb_per_spM.size(); ++i){
	    auto n_mb = n_mb_per_spM[i];
	    auto n_mt = n_mt_per_spM[i];
	    
	    mb_end_idx += n_mb;
	    mt_end_idx += n_mt;
	    
	    if (mb_end_idx > mb_idx){
		break;
	    }
	    mt_start_idx += n_mt;	    

	}
	
	/*
	for (auto n_mb : n_mb_per_spM){
	    //auto n_mid_top = n_doublets.second;
	    
	    end_idx += n_doublets;	    
	    if (end_idx > mb_idx){
		break;
	    }
	    start_idx += n_doublets;	    
	}        
	*/
	
	// iterate over mid-top doublets
	for (auto mt_it = mid_top_doublets_per_bin.begin()+mt_start_idx;
	     mt_it!= mid_top_doublets_per_bin.begin()+mt_end_idx ;
	     mt_it++){

	    auto lt = (*mt_it).lin;
	    
	    if (!triplet_finding_helper::isCompatible(spM, lb, lt, config,
						      iSinTheta2, scatteringInRegion2,
						      curvature, impact_parameter)){
		continue;
	    }

	    num_triplets_per_mid_bot++;
	    auto pos = atomicAdd(&num_triplets_per_bin,1);
	    if (pos>=triplets_per_bin.size()){
		num_triplets_per_bin = triplets_per_bin.size();
		continue;
	    }
	    
	    triplets_per_bin[pos] = triplet({mid_bot_doublet.sp2,
					     mid_bot_doublet.sp1,
					     (*mt_it).sp2,
					     curvature,
					     impact_parameter,
					     -impact_parameter*filter_config.impactWeightFactor,
					     lb.Zo
		});
	}
    }
    __syncthreads();
}
    
}// namespace cuda
}// namespace traccc

