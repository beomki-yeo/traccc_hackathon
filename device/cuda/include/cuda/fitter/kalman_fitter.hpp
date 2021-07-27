/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"

#include <cuda/fitter/gain_matrix_updater.cuh>
#include <edm/detail/transform_free_to_bound.hpp>
#include <edm/measurement.hpp>
#include <edm/track_state.hpp>
#include <geometry/surface.hpp>
#include <propagator/detail/covariance_engine.hpp>


namespace traccc {
namespace cuda {

template <typename propagator_t, typename updater_t, typename smoother_t>
class kalman_fitter {

    using kalman_navigator_t = typename propagator_t::navigator_t;
    using track_state_t = typename updater_t::track_state_type;

    public:
    kalman_fitter(propagator_t pPropagator)
        : m_propagator(std::move(pPropagator)) {}

 
    template <typename parameters_t>
    class actor {
        public:
        // ID of target surface
	std::vector<int>& target_surface_id;

	// Ref to the measurements
	traccc::host_measurement_collection& input_measurements;

	// Ref to the surfaces
	traccc::host_surface_collection& surfaces;

        /// Whether to consider multiple scattering.
        bool multiple_scattering = true;

        /// Whether to consider energy loss.
        bool energy_loss = true;

        /// Whether run reversed filtering
        bool reversed_filtering = false;

	actor(std::vector<int>& target_surface_id,
	      traccc::host_measurement_collection& input_measurements,
	      traccc::host_surface_collection& surfaces,
	      bool multiple_scattering = true,
	      bool energy_loss = true,
	      bool reversed_filtering = false) :
	    target_surface_id(target_surface_id),
	    input_measurements(input_measurements),
	    surfaces(surfaces),
	    multiple_scattering(multiple_scattering),
	    energy_loss(energy_loss),
	    reversed_filtering(reversed_filtering) {}

	template <typename propagator_state_t, typename stepper_t>
	track_state_t make_track_state(int target_surface_id, propagator_state_t& state, stepper_t& stepper) const {
	    // + Find the surface
	    if (target_surface_id >= surfaces.items.size())
		return;
	    traccc::surface &surf = surfaces.items.at(target_surface_id);

	    // Find matching measurement
	    auto res = std::find_if(input_measurements.begin(), input_measurements.end(),
				    [target_surface_id] (traccc::measurement &m) {
					return m.surface_id == target_surface_id;
				    }
		);
	    if (res == input_measurements.end())
		return;
	    traccc::measurement meas = *res;

	    // + Transport covariance to the surface
	    stepper.cov_transport(state);
	    
	    // + Pre Material effects (ignored for now)

	    // + Bind stepping state to the surface, get boundParams, jacobian, pathlength
	    // Note: this should be in the stepper
	    auto [boundParams, jacobian, pathLength] =
		traccc::detail::bound_state(
		    state.stepping.cov,
		    state.stepping.jacobian,
		    state.stepping.jac_transport,
		    state.stepping.derivative,
		    state.stepping.jac_to_global,
		    state.stepping.pars,
		    // state.stepping.covTransport,
		    false, // FIXME!!!!
		    state.stepping.path_accumulated,
		    target_surface_id,
		    surfaces);

	    // + Creates a track state, fill predicted part, measurement, projector
	    track_state_t tr_state;
	    tr_state.measurement() = std::move(meas);
	    tr_state.predicted() = std::move(boundParams);
	    tr_state.projector() = tr_state.measurement().projector();

	    // + Calibrate the measurement (ignored for now)
	    // + Outlier detection (ignored for now)

	    return tr_state;
	}

        template <typename propagator_state_t, typename stepper_t>
        void operator()(
	    std::vector<int>& target_surface_id,
	    std::vector<propagator_state_t>& state,
	    std::vector<stepper_t>& stepper) const {

	    if (target_surface_id.size() != stepper.size()) {
		std::cerr << "ERROR: kalman actor: target_surface_id.size() != stepper.size()" << std::endl;
		return;
	    }
	    if (state.size() != stepper.size()) {
		std::cerr << "ERROR: kalman actor: state.size() != stepper.size()" << std::endl;
		return;
	    }

	    vecmem::cuda::device_memory_resource host_mr;
	    traccc::host_collection<track_state_t> h_ts(
		{typename traccc::host_collection<track_state_t>::item_vector(state.size(), &host_mr)});

	    for (size_t i = 0; i < state.size(); i++) {
		h_ts.items.at(i) = make_track_state(target_surface_id.at(i), state.at(i), stepper.at(i));
	    }

	    vecmem::cuda::copy m_copy;
	    vecmem::cuda::device_memory_resource dev_mr;
	    traccc::device_collection<track_state_t> d_ts(
		{m_copy.to(vecmem::get_data(h_ts.items), dev_mr, vecmem::copy::type::host_to_device)});
	    

	    traccc::cuda::gain_matrix_updater<track_state_t> updater;
	    updater(d_ts);

	    m_copy(d_ts, h_ts, vecmem::copy::type::device_to_host);

	    for (size_t i = 0; i < state.size; i++) {
		h_ts.items.at(i).surface_id = target_surface_id.at(i);
		state.at(i).stepping = stepper.at(i).make_state(
		    h_ts.at(i).filtered(),
		    surfaces,
		    state.at(i).stepping.nav_dir
		    state.at(i).stepping.nav_dir,
		    state.at(i).stepping.step_size,
		    state.at(i).stepping.tolerance);
	    }
	}
    };

private:
    propagator_t m_propagator;

    template <typename parameters_t>
    class aborter {};
};

}  // namespace cuda
}  // namespace traccc
