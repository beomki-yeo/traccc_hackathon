/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/propagator/propagator.cuh>
#include <cuda/propagator/eigen_stepper.cuh>
#include <propagator/eigen_stepper.hpp>
#include <propagator/direct_navigator.hpp>
#include <cuda/utils/definitions.hpp>
#include <propagator/propagator_options.hpp>

namespace traccc {
namespace cuda {

// kernel declaration
template <typename propagator_state_t>
__global__ void rk4_kernel(
    collection_view<propagator_state_t> states_view);

// kernel declaration
template <typename propagator_state_t>
__global__ void cov_transport_kernel(
    collection_view<propagator_state_t> states_view);    


// explicit type instantiation    
using truth_propagator = typename traccc::cuda::propagator<traccc::eigen_stepper, traccc::direct_navigator>;
using void_propagator_options = typename traccc::propagator_options<void_actor, void_aborter>;
using void_multi_state = typename truth_propagator::multi_state< void_propagator_options >;    

template void traccc::cuda::eigen_stepper::rk4<void_multi_state>(void_multi_state& state);
    
template void traccc::cuda::eigen_stepper::cov_transport<void_multi_state>(void_multi_state& state);

template < typename propagator_state_t >    
void traccc::cuda::eigen_stepper::rk4(propagator_state_t& state) {
    
    auto states_view = get_data(state.states);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = states_view.items.size() / num_threads + 1;

    rk4_kernel< typename propagator_state_t::state_t> <<<num_blocks, num_threads>>>(states_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

template < typename propagator_state_t >
void traccc::cuda::eigen_stepper::cov_transport(propagator_state_t& state) {
    
    auto states_view = get_data(state.states);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = states_view.items.size() / num_threads + 1;
    
    // run the kernel
    cov_transport_kernel< typename propagator_state_t::state_t> 
        <<<num_blocks, num_threads>>>(states_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    
}

    
// kernel declaration
template <typename propagator_state_t>
__global__ void rk4_kernel(
			   collection_view<propagator_state_t> states_view){

    traccc::device_collection<propagator_state_t> states_device(
        {states_view.items});

    
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= states_device.items.size()) {
        return;
    }

    traccc::eigen_stepper::rk4(states_device.items.at(gid));    
}
    
// kernel declaration
template <typename propagator_state_t>
__global__ void cov_transport_kernel(
				     collection_view<propagator_state_t> states_view){

    traccc::device_collection<propagator_state_t> states_device(
        {states_view.items});

    
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= states_device.items.size()) {
        return;
    }

    traccc::eigen_stepper::cov_transport(states_device.items.at(gid));

}

    
}  // namespace cuda
}  // namespace traccc
