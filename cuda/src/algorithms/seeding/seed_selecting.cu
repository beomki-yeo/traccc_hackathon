/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cuda/algorithms/seeding/seed_selecting.cuh>
#include <cuda/utils/cuda_helper.cuh>
#include <algorithm>

namespace traccc{    
namespace cuda{

struct triplet_spM_ascending : public thrust::binary_function<triplet,triplet,bool>{
    __device__
    bool operator()(const triplet &lhs, const triplet &rhs) const {
	return lhs.sp2.sp_idx < rhs.sp2.sp_idx;
    }
}; 

    
struct triplet_weight_descending : public thrust::binary_function<triplet,triplet,bool>{
    __device__
    bool operator()(const triplet &lhs, const triplet &rhs) const {
	return lhs.weight > rhs.weight;
    }
}; 
    
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


    /*
    for (int i=0; i<triplet_container.headers.size(); ++i){
	thrust::sort(thrust::device,
		     triplet_container.items[i].begin(),
		     triplet_container.items[i].begin()+triplet_container.headers[i],
		     triplet_spM_ascending()
		     );
    }
    */

    
    unsigned int num_threads = WARP_SIZE*2; 
    unsigned int num_blocks = internal_sp_view.headers.m_size;
    unsigned int sh_mem = sizeof(triplet)*num_threads*filter_config.max_triplets_per_spM;
    
    seed_selecting_kernel
	<<< num_blocks,num_threads, sh_mem >>>(filter_config,
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

    size_t n_iter = num_compat_spM_per_bin/blockDim.x + 1;

    // zero initialization
    extern __shared__ triplet triplets_per_spM[];
    __syncthreads();
    
    for (size_t i_it = 0; i_it < n_iter; ++i_it){

	auto gid = i_it*blockDim.x + threadIdx.x;

	if (gid >= num_compat_spM_per_bin){
	    continue;
	}

	auto& spM_loc = doublet_counter_per_bin[gid].spM;	
	auto& spM_idx = spM_loc.sp_idx;
	auto& spM = internal_sp_per_bin[spM_idx];
	
	if (spM_idx >= doublet_counter_per_bin.size()) {
	    continue;
	}

	//size_t triplet_start_idx = 0;
	//size_t triplet_end_idx = 0;
	size_t n_triplets_per_spM = 0;
	
	bool found_spM = false;
	size_t stride = threadIdx.x*filter_config.max_triplets_per_spM;
	
	for (size_t i=0; i<num_triplets_per_bin; ++i){
	    auto& aTriplet = triplets_per_bin[i];
	    auto& spB_loc = aTriplet.sp1;
	    auto& spT_loc = aTriplet.sp3;
	    auto& spB = internal_sp_device.items[spB_loc.bin_idx][spB_loc.sp_idx];
	    auto& spT = internal_sp_device.items[spT_loc.bin_idx][spT_loc.sp_idx];
	    
	    if (spM_loc == aTriplet.sp2){
		if (!found_spM){		    
		    found_spM = true;		    
		}

		seed_selecting_helper::seed_weight(filter_config,
						   spB,
						   spT,
						   aTriplet.weight);

		if (!seed_selecting_helper::single_seed_cut(filter_config,
							    spB,
							    aTriplet.weight)){
		    continue;
		}

		
		triplets_per_spM[stride +n_triplets_per_spM] = aTriplet;	    
		n_triplets_per_spM++;

		if (n_triplets_per_spM == filter_config.max_triplets_per_spM){
		    //printf("hi \n");
		    break;
		}
	    }
	}

	if (!found_spM) {
	    continue;
	}
	
	/*
	for (size_t i=0; i<num_triplets_per_bin; ++i){
	    if (spM_loc == triplets_per_bin[i].sp2){
		if (!found_spM){		    
		    triplet_start_idx = i;
		    found_spM = true;		    
		}
		n_triplets_per_spM++;		
	    }
	}
	*/
	/*
	triplet_end_idx = triplet_start_idx + n_triplets_per_spM;

	size_t n_triplets_count = 0;
	
	for (size_t i = triplet_start_idx; i<triplet_end_idx; ++i){
	    auto& aTriplet = triplets_per_bin[i];	    
	    auto& spB_loc = aTriplet.sp1;
	    auto& spT_loc = aTriplet.sp3;
	    auto& spB = internal_sp_device.items[spB_loc.bin_idx][spB_loc.sp_idx];
	    auto& spT = internal_sp_device.items[spT_loc.bin_idx][spT_loc.sp_idx];
	    
	    //seed_selecting_helper::seed_weight(filter_config, spB, spT, aTriplet.weight);	    
	    //if (!seed_selecting_helper::single_seed_cut(filter_config, spB, aTriplet.weight)){
	    //aTriplet.weight = 1e-9; // quick prescription
	    //}
	    n_triplets_count++;
	}
	*/

	
	// sort the triplets per spM
	//sequential version of thrust sorting algorithm is used
	thrust::sort(thrust::seq,
		     triplets_per_spM+stride,
		     triplets_per_spM+stride+n_triplets_per_spM,
		     triplet_weight_descending());
	
	
	size_t n_seeds_per_spM = 0;

	//n_triplets_count = fmin(n_triplets_count,5);	
	
	//for (size_t i=triplet_start_idx; i<triplet_start_idx+n_triplets_count; ++i){
	for (size_t i=stride; i<stride+n_triplets_per_spM; ++i){
	    
	    auto& aTriplet = triplets_per_spM[i];	    
	    auto& spB_loc = aTriplet.sp1;
	    auto& spT_loc = aTriplet.sp3;
	    auto& spB = internal_sp_device.items[spB_loc.bin_idx][spB_loc.sp_idx];
	    auto& spT = internal_sp_device.items[spT_loc.bin_idx][spT_loc.sp_idx];
	    	    
	    if (n_seeds_per_spM >= filter_config.maxSeedsPerSpM-1){
		break;
	    }	    
	    
	    if (seed_selecting_helper::cut_per_middle_sp(filter_config, spB, aTriplet.weight) ||
		n_seeds_per_spM==0){
		
		auto pos = atomicAdd(&num_seeds, 1);
		
		if (pos >= seeds.size()){
		    break;
		}		
		n_seeds_per_spM++;
		seeds[pos] = seed({spB.m_sp,
				   spM.m_sp,
				   spT.m_sp,
				   aTriplet.weight,
				   aTriplet.z_vertex});		

	    }
	    
	}
	
    }
}

}// namespace cuda
}// namespace traccc
