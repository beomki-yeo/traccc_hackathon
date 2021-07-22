/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/fitter/gain_matrix_updater.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

// kernel declareation

template <typename track_state_t>
__global__ void update_kernel(
    track_state_collection_view<track_state_t> track_states_view);

template class gain_matrix_updater<
    track_state<measurement, bound_track_parameters>>;

// implementation of kalman gain matrix update function
template <typename track_state_t>
void gain_matrix_updater<track_state_t>::operator()(
    host_track_state_collection<track_state_t>& track_states,
    vecmem::memory_resource* resource) {

    auto track_state_view = get_data(track_states, resource);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = track_state_view.items.size() / num_threads + 1;

    // run the kernel
    update_kernel<track_state_t><<<num_blocks, num_threads>>>(track_state_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

template <typename track_state_t>
__global__ void update_kernel(
    track_state_collection_view<track_state_t> track_states_view) {

    device_track_state_collection<track_state_t> track_states_device(
        {track_states_view.items});

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= track_states_device.items.size()) {
        return;
    }

    gain_matrix_updater_impl<track_state_t>::update(
        track_states_device.items.at(gid));
}

}  // namespace cuda
}  // namespace traccc
