/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/propagator/eigen_stepper.cuh>
#include <propagator/propagator_options.hpp>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

using state = traccc::eigen_stepper::state;
__global__ void stepper_kernel(traccc::collection_view<state> stepper_state_view);

// Kernel for cov transport
template <typename propagator_options_t>
__global__ void cov_transport_kernel(
    collection_view<state> states_view, collection_view<propagator_options_t> options_view);

// Reserved to Xiangyang

bool traccc::cuda::eigen_stepper::rk4(host_collection<state>& states) {
    auto stepper_state_view = get_data(states);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = stepper_state_view.items.size() / num_threads + 1;

    stepper_kernel<<<num_blocks, num_threads>>>(stepper_state_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return true;
}

__global__ void stepper_kernel(traccc::collection_view<state> stepper_state_view)
{
    traccc::device_collection<state> stepper_states_device(
        {stepper_state_view.items});

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= stepper_states_device.items.size()) {
        return;
    }

    traccc::eigen_stepper::rk4(stepper_states_device.items.at(gid));  
}


template void traccc::cuda::eigen_stepper::cov_transport<
    propagator_options<void_actor, void_aborter> >(
    host_collection<state>& state,
    host_collection<propagator_options<void_actor, void_aborter> >& options);

// Reserved to Johannes
template <typename propagator_options_t>
void traccc::cuda::eigen_stepper::cov_transport(
    host_collection<state>& state,
    host_collection<propagator_options_t>& options) {

    vecmem::cuda::managed_memory_resource mng_mr;
    auto state_view = get_data(state, &mng_mr);
    auto options_view = get_data(options, &mng_mr);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = state_view.items.size() / num_threads + 1;

    // run the kernel
    cov_transport_kernel<propagator_options_t><<<num_blocks, num_threads>>>(state_view, options_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

template <typename propagator_options_t>
__global__ void traccc::cuda::cov_transport_kernel(
    collection_view<state> states_view, collection_view<propagator_options_t> options_view) {
    
    //access collection
    device_collection<state> states_device({states_view.items});
    device_collection<propagator_options_t> options_device({options_view.items});

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= states_device.items.size()) {
        return;
    }

    //run cov_transport function
    traccc::eigen_stepper::cov_transport(
        states_device.items.at(gid), options_device.items.at(gid).mass);
}

}  // namespace cuda
}  // namespace traccc

