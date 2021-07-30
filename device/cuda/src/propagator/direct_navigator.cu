/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/propagator/propagator.cuh>
#include <cuda/propagator/direct_navigator.cuh>
#include <propagator/eigen_stepper.hpp>
#include <propagator/direct_navigator.hpp>
#include <cuda/utils/definitions.hpp>
#include <propagator/propagator_options.hpp>

namespace traccc {
namespace cuda {


// kernel declaration
template <typename propagator_state_t, typename surface_t>
__global__ void status_kernel(collection_view<propagator_state_t> states_view,
			      collection_view<surface_t> surfaces_view);    

    
// explicit type instantiation    
using truth_propagator = typename traccc::cuda::propagator<traccc::eigen_stepper, traccc::direct_navigator>;
using void_propagator_options = typename traccc::propagator_options<void_actor, void_aborter>;
using void_multi_state = typename truth_propagator::multi_state< void_propagator_options >;    

template void traccc::cuda::direct_navigator::status<void_multi_state, surface>(void_multi_state& state, host_collection< surface >& surfaces);


template <typename propagator_state_t, typename surface_t>
void traccc::cuda::direct_navigator::status(propagator_state_t& state,
					    host_collection<surface_t>& surfaces){

    auto states_view = get_data(state.states);
    auto surfaces_view = get_data(surfaces);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = states_view.items.size() / num_threads + 1;
    
    // run the kernel
    status_kernel< typename propagator_state_t::state_t, surface_t > 
        <<<num_blocks, num_threads>>>(states_view, surfaces_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    
}

template <typename propagator_state_t, typename surface_t>
__global__ void status_kernel(collection_view<propagator_state_t> states_view,
			      collection_view<surface_t> surfaces_view){

    device_collection<propagator_state_t> states_device({states_view.items});
    device_collection<surface_t> surfaces_device({surfaces_view.items});
    
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= states_device.items.size()) {
        return;
    }

    traccc::direct_navigator::status(states_device.items.at(gid),
                                     &surfaces_device.items.at(0));
    
}
    
}  // namespace cuda
}  // namespace traccc
