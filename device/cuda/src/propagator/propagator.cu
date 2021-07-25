/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/propagator/propagator.cuh>
#include <cuda/utils/definitions.hpp>
#include <edm/track_parameters.hpp>
#include <propagator/propagator_options.hpp>
#include <propagator/eigen_stepper.hpp>
#include <propagator/direct_navigator.hpp>

namespace traccc {
namespace cuda {
    
// kernel declaration    
template < typename propagator_options_t, typename stepper_t,
	   typename navigator_t, typename surface_t>
__global__ void propagate_kernel(
		      collection_view< propagator_options_t > options_view,
		      collection_view< typename stepper_t::state > stepping_view,
		      collection_view< typename navigator_t::state > navigator_view,
		      collection_view< surface_t > surfaces_view);   
    
    
template 
void propagator< traccc::eigen_stepper, traccc::direct_navigator >::propagate< traccc::propagator_options<void_actor, void_aborter>, surface >(
     host_collection<typename traccc::propagator_options<void_actor, void_aborter>>& options,
     host_collection<typename traccc::eigen_stepper::state>& stepping,
     host_collection<typename traccc::direct_navigator::state>& navigation,
     host_collection<surface>& surfaces,
     vecmem::memory_resource* resource);
    
// definition
template <typename stepper_t, typename navigator_t >
template <typename propagator_options_t, typename surface_t> 
void traccc::cuda::propagator<stepper_t, navigator_t>::propagate(
    host_collection<propagator_options_t>& options,
    host_collection<typename stepper_t::state>& stepping,
    host_collection<typename navigator_t::state>& navigation,
    host_collection<surface_t>& surfaces,
    vecmem::memory_resource* resource) {

    auto options_view = get_data(options, resource);
    auto stepping_view = get_data(stepping, resource);
    auto navigation_view = get_data(navigation, resource);
    auto surfaces_view = get_data(surfaces, resource);

    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = options_view.items.size() / num_threads + 1;
    
    // run the kernel    
    propagate_kernel<propagator_options_t, stepper_t, navigator_t, surface_t>
        <<<num_blocks, num_threads>>>(options_view, stepping_view, navigation_view, surfaces_view);
    
    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    
}    
    
// kernel implementation
template < typename propagator_options_t, typename stepper_t,
	   typename navigator_t, typename surface_t>
__global__ void propagate_kernel(
		      collection_view< propagator_options_t > options_view,
		      collection_view< typename stepper_t::state > stepping_view,
		      collection_view< typename navigator_t::state > navigator_view,
		      collection_view< surface_t > surfaces_view){
    device_collection< propagator_options_t > options_device({options_view.items});
    device_collection< typename stepper_t::state > stepping_device({stepping_view.items});
    device_collection< typename navigator_t::state > navigator_device({navigator_view.items});
    device_collection< surface_t > surfaces_device({surfaces_view.items});

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (gid >= options_device.items.size()) {
        return;
    }

    stepper_t stepper;
    navigator_t navigator;
    
    traccc::propagator prop(stepper,navigator);

    prop.propagate(options_device.items.at(gid),
		   stepping_device.items.at(gid),
		   navigator_device.items.at(gid),
		   &surfaces_device.items.at(0));
   
} 
    
}  // namespace cuda
}  // namespace traccc
    
