/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "cuda/propagator/detail/eigen_stepper_impl.cuh"

namespace traccc {
namespace cuda {

template <typename stepper_state_t>
void eigen_stepper_impl::cov_transport(host_collection<stepper_state_t>& state,
                                       const Acts::ActsScalar mass) {}

}  // namespace cuda
}  // namespace traccc
