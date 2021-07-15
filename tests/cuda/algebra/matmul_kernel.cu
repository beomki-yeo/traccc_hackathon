/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda_runtime.h>
#include "matmul_kernel.cuh"

__global__ void matmul_kernel(
    vecmem::data::vector_view< Acts::BoundSymMatrix > A_view,
    vecmem::data::vector_view< Acts::BoundSymMatrix > B_view,
    vecmem::data::vector_view< Acts::BoundSymMatrix > C_view);

__global__ void matmul_kernel(
    vecmem::data::vector_view< Acts::FreeSymMatrix > A_view,
    vecmem::data::vector_view< Acts::FreeSymMatrix > B_view,
    vecmem::data::vector_view< Acts::FreeSymMatrix > C_view);

void matmul(int n_matrix,
	    vecmem::data::vector_view< Acts::BoundSymMatrix > A_view,
	    vecmem::data::vector_view< Acts::BoundSymMatrix > B_view,
	    vecmem::data::vector_view< Acts::BoundSymMatrix > C_view,
	    float& time){

    unsigned int num_threads = WARP_SIZE*2;
    unsigned int num_blocks = n_matrix/num_threads + 1;

    //--TIME--------------------------------------
    float elapsed=0;
    cudaEvent_t start, stop;    
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));
    CUDA_ERROR_CHECK(cudaEventRecord(start, 0));
    //--------------------------------------------
    
    matmul_kernel<<< num_blocks, num_threads >>>(A_view, B_view, C_view);

    // cuda error check    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    
    //--TIME--------------------------------------
    CUDA_ERROR_CHECK(cudaEventRecord(stop, 0));
    CUDA_ERROR_CHECK(cudaEventSynchronize (stop) );    
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsed, start, stop) );    
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
    elapsed*=0.001; // ms->sec
    time += elapsed;
    //--------------------------------------------
    
}

void matmul(int n_matrix,
	    vecmem::data::vector_view< Acts::FreeSymMatrix > A_view,
	    vecmem::data::vector_view< Acts::FreeSymMatrix > B_view,
	    vecmem::data::vector_view< Acts::FreeSymMatrix > C_view,
	    float& time){

    unsigned int num_threads = WARP_SIZE*2;
    unsigned int num_blocks = n_matrix/num_threads + 1;

    //--TIME--------------------------------------
    float elapsed=0;
    cudaEvent_t start, stop;    
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));
    CUDA_ERROR_CHECK(cudaEventRecord(start, 0));
    //--------------------------------------------
    
    matmul_kernel<<< num_blocks, num_threads >>>(A_view, B_view, C_view);

    //--TIME--------------------------------------
    CUDA_ERROR_CHECK(cudaEventRecord(stop, 0));
    CUDA_ERROR_CHECK(cudaEventSynchronize (stop) );    
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsed, start, stop) );    
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));
    elapsed*=0.001; // ms->sec
    time += elapsed;
    //--------------------------------------------
    
    // cuda error check    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


__global__ void matmul_kernel(
    vecmem::data::vector_view< Acts::BoundSymMatrix > A_view,
    vecmem::data::vector_view< Acts::BoundSymMatrix > B_view,
    vecmem::data::vector_view< Acts::BoundSymMatrix > C_view){
    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    vecmem::device_vector< Acts::BoundSymMatrix > A_dev(A_view);
    vecmem::device_vector< Acts::BoundSymMatrix > B_dev(B_view);
    vecmem::device_vector< Acts::BoundSymMatrix > C_dev(C_view);

    if (gid >= A_dev.size()){
	return;
    }
        
    C_dev.at(gid) = A_dev.at(gid)*B_dev.at(gid);
}

__global__ void matmul_kernel(
    vecmem::data::vector_view< Acts::FreeSymMatrix > A_view,
    vecmem::data::vector_view< Acts::FreeSymMatrix > B_view,
    vecmem::data::vector_view< Acts::FreeSymMatrix > C_view){

    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    vecmem::device_vector< Acts::FreeSymMatrix > A_dev(A_view);
    vecmem::device_vector< Acts::FreeSymMatrix > B_dev(B_view);
    vecmem::device_vector< Acts::FreeSymMatrix > C_dev(C_view);

    if (gid >= A_dev.size()){
	return;
    }
        
    C_dev.at(gid) = A_dev.at(gid)*B_dev.at(gid);
}
