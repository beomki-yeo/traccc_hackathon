/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/propagator/eigen_stepper.cuh>

namespace traccc {    
namespace cuda {

// Reserved to Xiangyang
bool traccc::cuda::eigen_stepper::rk4(host_state_collection& state){
    return true;
}

// Reserved to Johannes    
void traccc::cuda::eigen_stepper::cov_transport(host_state_collection& state, const Acts::ActsScalar mass){

    
    
}
    
} // namespace cuda    
} // namespace traccc
    
