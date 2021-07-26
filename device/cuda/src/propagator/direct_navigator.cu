/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/propagator/direct_navigator.cuh>
#include <cuda/propagator/eigen_stepper.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

template <typename navigator_state_t, typename stepper_state_t,
          typename surface_t>
__global__ void status_kernel(
    collection_view<navigator_state_t> navigator_state_view,
    collection_view<stepper_state_t> stepper_state_view,
    collection_view<surface_t> surface_view);

// explicit instantiation
template bool direct_navigator::status<typename eigen_stepper::state, surface>(
    host_collection<state>& navigator_state,
    host_collection<typename eigen_stepper::state>& stepper_state,
    host_collection<surface>& surfaces, vecmem::memory_resource* resource);

// definition
template <typename stepper_state_t, typename surface_t>
bool direct_navigator::status(host_collection<state>& navigator_state,
                              host_collection<stepper_state_t>& stepper_state,
                              host_collection<surface_t>& surfaces,
                              vecmem::memory_resource* resource) {

    auto navigator_state_view = get_data(navigator_state, resource);
    auto stepper_state_view = get_data(stepper_state, resource);
    auto surfaces_view = get_data(surfaces, resource);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks =
        navigator_state_view.items.size() / num_threads + 1;

    // run the kernel
    status_kernel<state, stepper_state_t, surface_t>
        <<<num_blocks, num_threads>>>(navigator_state_view, stepper_state_view,
                                      surfaces_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return true;
}

template <typename navigator_state_t, typename stepper_state_t,
          typename surface_t>
__global__ void status_kernel(
    collection_view<navigator_state_t> navigator_state_view,
    collection_view<stepper_state_t> stepper_state_view,
    collection_view<surface_t> surface_view) {

    device_collection<navigator_state_t> navigator_state_device(
        {navigator_state_view.items});
    device_collection<stepper_state_t> stepper_state_device(
        {stepper_state_view.items});
    device_collection<surface_t> surface_device({surface_view.items});

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= navigator_state_device.items.size()) {
        return;
    }

    traccc::direct_navigator::status(navigator_state_device.items.at(gid),
                                     stepper_state_device.items.at(gid),
                                     &surface_device.items.at(0));
}

}  // namespace cuda
}  // namespace traccc
