/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/utils/warming_up.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

__global__ void warming_up_kernel();
    
void warming_up(){
       
    warming_up_kernel<<<100, 512>>>();
    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

}

__global__ void warming_up_kernel(){

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;

  return;
}
    
}  // namespace cuda
}  // namespace traccc

