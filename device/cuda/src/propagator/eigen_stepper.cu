/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/propagator/eigen_stepper.cuh>
#include <propagator/propagator_options.hpp>

namespace traccc {
namespace cuda {

// Reserved to Xiangyang
bool traccc::cuda::eigen_stepper::rk4(host_collection<state>& state) {
    return true;
}

template 
void traccc::cuda::eigen_stepper::cov_transport< propagator_options < void_actor, void_aborter > >(
     host_collection< state >& state,
     host_collection< propagator_options < void_actor, void_aborter > >& options);
    
// Reserved to Johannes
template<typename propagator_options_t>    
void traccc::cuda::eigen_stepper::cov_transport(host_collection<state>& state,
						host_collection<propagator_options_t>& options) {}

}  // namespace cuda
}  // namespace traccc
