/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/track_parameters.hpp>
#include <edm/track_state.hpp>
#include <edm/measurement.hpp>
#include <fitter/gain_matrix_smoother_impl.hpp>

namespace traccc {
namespace cuda {

template <typename track_state_t >    
class gain_matrix_smoother{
public:
    
    gain_matrix_smoother() = default;

    ~gain_matrix_smoother() = default;

    // declaration of kalman gain matrix update function
    void operator()(host_track_state_collection< track_state_t >& track_states, vecmem::memory_resource* resource);
    
private:   
};


}  // namespace cuda
}  // namespace traccc

    
