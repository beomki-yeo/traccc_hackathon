/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/detail/transform_free_to_bound.hpp>
#include <edm/measurement.hpp>
#include <geometry/surface.hpp>
#include <propagator/detail/covariance_engine.hpp>

namespace traccc {

template <typename propagator_t, typename updater_t, typename smoother_t>
class kalman_fitter {

    using kalman_navigator_t = typename propagator_t::navigator_t;

    public:
    kalman_fitter(propagator_t pPropagator)
        : m_propagator(std::move(pPropagator)) {}

    template <typename parameters_t>
    class actor {
        public:
        // ID of target surface
        int target_surface_id;

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

	actor(int target_surface_id,
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
        void operator()(propagator_state_t& state,
                        const stepper_t& stepper) const {

	    // + Find the surface
	    if (target_surface_id >= surfaces.items.size())
		return;
	    traccc::surface &surf = surfaces.items.at(target_surface_id);

	    // Find matching measurement
	    auto res = std::find_if(input_measurements.begin(), input_measurements.end(),
			 [this] (traccc::measurement &m) {
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
	    typename updater_t::track_state_type tr_state;
	    tr_state.measurement() = std::move(meas);
	    tr_state.predicted() = std::move(boundParams);
	    tr_state.projector() = tr_state.measurement().projector();

	    // + Calibrate the measurement (ignored for now)
	    // + Outlier detection (ignored for now)

            // + If not outlier, run the updator which fills the filtered state
	    updater_t updater;
	    updater(tr_state);
	    tr_state.filtered().surface_id = target_surface_id;

	    // + If not Outlier, Update the stepping state
	    state.stepping = stepper.make_state(
		tr_state.filtered(),
		surfaces,
		state.stepping.nav_dir,
		state.stepping.step_size,
		state.stepping.tolerance);

	    // + Post Material effects (ignored for now)
	}
    };

private:
    propagator_t m_propagator;



    template <typename parameters_t>
    class aborter {};
};

}  // namespace traccc
