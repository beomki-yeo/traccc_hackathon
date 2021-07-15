/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/track_parameters.hpp>

// Acts
#include <Acts/Definitions/TrackParametrization.hpp>

namespace traccc {

template <typename track_state_t>
class gain_matrix_updater{
public:

    gain_matrix_updater() = default;

    ~gain_matrix_updater() = default;

    void update(track_state_t& tr_state) {

	// measurement position and covariance
	const auto& meas_pos = tr_state.measurement().get_local();
	const auto& meas_cov = tr_state.measurement().get_variance();	

	// predicted states 
	const auto& pred_vec = tr_state.predicted().vector();
	const auto& pred_cov = tr_state.predicted().covariance();

	// projection matrix
	auto projection = tr_state.projector();

	//// write the kalman gain updating algorithm below
	
    }
};


} // namespace traccc
