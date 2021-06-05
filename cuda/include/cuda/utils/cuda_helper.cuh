/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/utils/definitions.hpp>

namespace traccc{
namespace cuda{    

struct cuda_helper{

template< typename T >    
static  __device__
void reduce_sum(size_t block_size, size_t tid, T* array){
    
    array[tid] += __shfl_down_sync(0xFFFFFFFF,array[tid],WARP_SIZE/2, WARP_SIZE);
    array[tid] += __shfl_down_sync(0xFFFFFFFF,array[tid],WARP_SIZE/4, WARP_SIZE/2);
    array[tid] += __shfl_down_sync(0xFFFFFFFF,array[tid],WARP_SIZE/8, WARP_SIZE/4);
    array[tid] += __shfl_down_sync(0xFFFFFFFF,array[tid],WARP_SIZE/16,WARP_SIZE/8);
    array[tid] += __shfl_down_sync(0xFFFFFFFF,array[tid],WARP_SIZE/32,WARP_SIZE/16);
    
    __syncthreads();
    
    if (tid == 0){
	for (int i=1; i<block_size/WARP_SIZE; i++){
	    array[tid] += array[i*WARP_SIZE];
	}
    }

    // deprecated version
    /*
    for(size_t i=1; i<block_size; i=i*2){
	if(tid % (2*i) == 0) {
	    array[tid] += array[tid + i];
	}
	__syncthreads();
    } 
    */
}

};

} // namespace cuda
} // namespace traccc
