/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cuda/algorithms/seeding/seed_selecting.cuh>
#include <cuda/utils/cuda_helper.cuh>

namespace traccc {
namespace cuda {

struct triplet_weight_descending
    : public thrust::binary_function<triplet, triplet, bool> {
    __device__ bool operator()(const triplet& lhs, const triplet& rhs) const {
        if (lhs.weight != rhs.weight) {
            return lhs.weight > rhs.weight;
        } else {
            return fabs(lhs.z_vertex) < fabs(rhs.z_vertex);
        }
    }
};

__device__ static bool triplet_weight_compare(const triplet& lhs,
                                              const triplet& rhs) {
    if (lhs.weight != rhs.weight) {
        return lhs.weight < rhs.weight;
    } else {
        return fabs(lhs.z_vertex) > fabs(rhs.z_vertex);
    }
}

__global__ void seed_selecting_kernel(
    const seedfilter_config filter_config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view, seed_container_view seed_view);

void seed_selecting(const seedfilter_config& filter_config,
                    host_internal_spacepoint_container& internal_sp_container,
                    host_doublet_counter_container& doublet_counter_container,
                    host_triplet_counter_container& triplet_counter_container,
                    host_triplet_container& triplet_container,
                    host_seed_container& seed_container,
                    vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);

    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);
    auto triplet_counter_container_view =
        get_data(triplet_counter_container, resource);
    auto triplet_container_view = get_data(triplet_container, resource);
    auto seed_container_view = get_data(seed_container, resource);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = 0;
    for (size_t i=0; i<internal_sp_view.headers.m_size; ++i){
	num_blocks += triplet_counter_container.headers[i] / num_threads +1;
    }
    
    unsigned int sh_mem =
        sizeof(triplet) * num_threads * filter_config.max_triplets_per_spM;

    seed_selecting_kernel<<<num_blocks, num_threads, sh_mem>>>(
        filter_config, internal_sp_view, doublet_counter_container_view,
        triplet_counter_container_view, triplet_container_view,
        seed_container_view);

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void seed_selecting_kernel(
    const seedfilter_config filter_config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view, seed_container_view seed_view) {
    device_internal_spacepoint_container internal_sp_device(
        {internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});
    device_triplet_counter_container triplet_counter_device(
        {triplet_counter_view.headers, triplet_counter_view.items});
    device_triplet_container triplet_device(
        {triplet_view.headers, triplet_view.items});
    device_seed_container seed_device({seed_view.headers, seed_view.items});

    unsigned int n_bins = internal_sp_device.headers.size();
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;

    cuda_helper::get_bin_idx(n_bins,
			     triplet_counter_device,
			     bin_idx,
			     ref_block_idx);
    
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);
    auto& num_compat_spM_per_bin =
        doublet_counter_device.headers.at(bin_idx);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(bin_idx);

    auto& num_compat_mb_per_bin = triplet_counter_device.headers.at(bin_idx);
    auto triplet_counter_per_bin = triplet_counter_device.items.at(bin_idx);

    auto& num_triplets_per_bin = triplet_device.headers.at(bin_idx);
    auto triplets_per_bin = triplet_device.items.at(bin_idx);

    auto& num_seeds = seed_device.headers.at(0);
    auto seeds = seed_device.items.at(0);

    extern __shared__ triplet triplets_per_spM[];
    __syncthreads();

    auto gid = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;
    
    if (gid >= num_compat_spM_per_bin) {
	return;
    }
    
    auto& spM_loc = doublet_counter_per_bin[gid].spM;
    auto& spM_idx = spM_loc.sp_idx;
    auto& spM = internal_sp_per_bin[spM_idx];
    
    if (spM_idx >= doublet_counter_per_bin.size()) {
	return;
    }
    
    size_t n_triplets_per_spM = 0;
    
    size_t stride = threadIdx.x * filter_config.max_triplets_per_spM;
    
    for (size_t i = 0; i < num_triplets_per_bin; ++i) {
	auto& aTriplet = triplets_per_bin[i];
	auto& spB_loc = aTriplet.sp1;
	auto& spT_loc = aTriplet.sp3;
	auto& spB =
	    internal_sp_device.items[spB_loc.bin_idx][spB_loc.sp_idx];
	auto& spT =
	    internal_sp_device.items[spT_loc.bin_idx][spT_loc.sp_idx];
	
	if (spM_loc == aTriplet.sp2) {
	    seed_selecting_helper::seed_weight(filter_config, spB, spT,
					       aTriplet.weight);
	    
	    if (!seed_selecting_helper::single_seed_cut(filter_config, spB,
							aTriplet.weight)) {
		continue;
	    }
	    
	    if (n_triplets_per_spM >= filter_config.max_triplets_per_spM) {
		int begin_idx = stride;
		int end_idx = stride + filter_config.max_triplets_per_spM;
		
		// Note: min_index method gives a result different
		//       from sorting method when there are the cases where
		//       weight & z_vertex are same.
		//
		//       So min_index method reduces seed matching ratio
		//       since the cpu version is using sorting method.
		//
		//       But that doesn't mean min_index method
		//       is wrong of course
		//
		//       Let's not be so obsessed about achieving
		//       perfectly same result :))))))))
		
		int min_index =
		    std::min_element(triplets_per_spM + begin_idx,
				     triplets_per_spM + end_idx,
				     triplet_weight_compare) -
		    triplets_per_spM;
		
		auto& min_weight = triplets_per_spM[min_index].weight;
		
		if (aTriplet.weight > min_weight) {
		    triplets_per_spM[min_index] = aTriplet;
		}
		
		// (deprecated) sorting method -> good for seed matching ratio but slow
		/*
		  thrust::sort(thrust::seq,
		  triplets_per_spM+begin_idx,
		  triplets_per_spM+end_idx,
		  triplet_weight_descending());
		  
		  if (aTriplet.weight >= triplets_per_spM[end_idx-1].weight){
		  triplets_per_spM[end_idx-1] = aTriplet;
		  }
		*/
	    }
	    
	    else if (n_triplets_per_spM <
		     filter_config.max_triplets_per_spM) {
		triplets_per_spM[stride + n_triplets_per_spM] = aTriplet;
		n_triplets_per_spM++;
	    }
	}
    }
    
    // sort the triplets per spM
    // sequential version of thrust sorting algorithm is used
    thrust::sort(thrust::seq, triplets_per_spM + stride,
		 triplets_per_spM + stride + n_triplets_per_spM,
		 triplet_weight_descending());
    
    size_t n_seeds_per_spM = 0;
    
    for (size_t i = stride; i < stride + n_triplets_per_spM; ++i) {
	auto& aTriplet = triplets_per_spM[i];
	auto& spB_loc = aTriplet.sp1;
	auto& spT_loc = aTriplet.sp3;
	auto& spB =
	    internal_sp_device.items[spB_loc.bin_idx][spB_loc.sp_idx];
	auto& spT =
	    internal_sp_device.items[spT_loc.bin_idx][spT_loc.sp_idx];
	
	if (n_seeds_per_spM >= filter_config.maxSeedsPerSpM + 1) {
	    break;
	}
	
	if (seed_selecting_helper::cut_per_middle_sp(filter_config, spB,
						     aTriplet.weight) ||
	    n_seeds_per_spM == 0) {
	    auto pos = atomicAdd(&num_seeds, 1);
	    
	    if (pos >= seeds.size()) {
		break;
	    }
	    n_seeds_per_spM++;
	    
	    seeds[pos] = seed({spB.m_sp, spM.m_sp, spT.m_sp,
			       aTriplet.weight, aTriplet.z_vertex});
	}
    }
}

}  // namespace cuda
}  // namespace traccc
