/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

template < typename track_state_t >
struct gain_matrix_updater_impl{

    // type definition
    using param_vec_t = typename track_state_t::param_vec_t;
    using param_cov_t = typename track_state_t::param_cov_t;
    using meas_vec_t = typename track_state_t::meas_vec_t;
    using meas_cov_t = typename track_state_t::meas_cov_t;
    using projector_t = typename track_state_t::projector_t;

    static __CUDA_HOST_DEVICE__ void update(track_state_t& tr_state){

	auto& meas = tr_state.measurement();
	
	// measurement position and covariance
	const auto& meas_local = meas.get_local();
	const auto& meas_variance = meas.get_variance();	
	
	meas_vec_t meas_vec;
	meas_cov_t meas_cov;
	meas_vec << meas_local[0], meas_local[1];
	meas_cov << meas_variance[0], 0, 0, meas_variance[1];
	    	
	// predicted states
	// (6x1)
	const auto& pred_vec = tr_state.predicted().vector();
	// (6x6)
	const auto& pred_cov = tr_state.predicted().covariance();

	// filtered states
	// (6x1)
	auto& filtered_vec = tr_state.filtered().vector();
	// (6x6)
	auto& filtered_cov = tr_state.filtered().covariance();

	// projector matrix
	// (2x6)
	const auto& H = tr_state.projector(); 
	
	// (2x6) * (6x6) * (6x2) + (2x2)
	meas_cov_t cov = H * pred_cov * H.transpose() + meas_cov;		
		
	// (6x6) * (6x2) * (2x2)
	auto K = pred_cov * H.transpose() * cov.inverse();
	
	// (2x1)
	meas_vec_t residual = meas.get_residual(pred_vec);
	
	// (6x2) * (2x1)
	param_vec_t gain = K * residual;
	
	// (6x1) + (6x1)
	filtered_vec = pred_vec + gain;
		
	// (6x6) - (6x2) * (2x6)
	param_cov_t C = param_cov_t::Identity() - param_cov_t(K * H);
	
	// (6x6) * (6x6)
	tr_state.filtered().covariance() = C * pred_cov;			
    }    
};
    
} // namespace traccc
