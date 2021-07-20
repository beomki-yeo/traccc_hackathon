/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "Acts/Definitions/Algebra.hpp"

namespace traccc {

struct eigen_stepper_impl {

    template <typename stepper_state_t>
    static __CUDA_HOST_DEVICE__ void cov_transport(const stepper_state_t& state,
						   Acts::FreeMatrix& D){
	//D = Acts::FreeMatrix::Identity();
	//auto dir = state.dir;
	//auto qop = state.q / state.p;	
    }
};
    
}  // namespace traccc
    
