/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda/fitter/detail/gain_matrix_updater_impl.cuh>
#include <edm/track_parameters.hpp>

// Acts
#include <Acts/Definitions/TrackParametrization.hpp>

namespace traccc {
namespace cuda {
    
template<typename scalar_t,
	 int meas_dim,
	 int params_dim,
	 int batch_size>
class gain_matrix_updater{
public:
        
    gain_matrix_updater() = default;

    ~gain_matrix_updater() = default;

    template <typename track_state_device_t>
    void update(track_state_device_t& track_states) {
	scalar_t* meas_array[batch_size];
	scalar_t* proj_array[batch_size];
	scalar_t* pred_vector_array[batch_size];
	scalar_t* pred_cov_array[batch_size];
	
	for (unsigned int i_b=0; i_b<batch_size; i_b++){
	    auto& tr_state = track_states.ptr()[i_b];	    
	    meas_array[i_b] = &(tr_state.measurement().get_local()[0]);
	    proj_array[i_b] = &(tr_state.projector()(0));	   
	    pred_vector_array[i_b] = &(tr_state.predicted().vector()(0));
	    pred_cov_array[i_b] = &(tr_state.predicted().covariance()(0));	    
	}
	
	m_impl.update(meas_array, proj_array, pred_vector_array, pred_cov_array);
    }
    
private:
    gain_matrix_updater_impl<scalar_t, meas_dim, params_dim, batch_size> m_impl;    
};


}  // namespace cuda
}  // namespace traccc

