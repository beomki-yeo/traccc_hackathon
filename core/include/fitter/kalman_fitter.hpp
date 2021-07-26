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

	    // SOME POINTERS
	    //
	    // -> Follow the ACTS filter code
	    // -> BoundState from the stepper
	    // -> Covariance transport
	    // -> Fill track state proxy in the same way
	    // Basically look into what is done up to the m_updater call, and how the result is treated

	    // Find matching measurement
	    auto res = std::find_if(input_measurements.begin(), input_measurements.end(),
			 [this] (traccc::measurement &m) {
			     return m.surface_id == target_surface_id;
			 }
	    );
	    if (res == input_measurements.end())
		return;


	    traccc::measurement &meas = *res;

	    // Find the surface itself
	    if (target_surface_id >= surfaces.items.size())
		return;
	    traccc::surface &surf = surfaces.items.at(target_surface_id);

	    //state: measurement, predicted vector + covariance
	    typename updater_t::track_state_type tr_state;
	    tr_state.measurement() = meas;
	    tr_state.predicted().covariance() = state.stepping.cov;

	    // FIXME this is predefined somewhere
	    tr_state.projector() = Acts::ActsMatrix<2,6>::Zero();
	    tr_state.projector()(0,0) = 1;
	    tr_state.projector()(1,1) = 1;

	    tr_state.predicted().vector() =
		traccc::detail::transform_free_to_bound_parameters(state.stepping.pars, surf);

	    updater_t updater;
	    updater(tr_state);

	    state.stepping.pars = traccc::detail::transform_bound_to_free_parameters(tr_state.filtered().vector(), surf);
	    state.stepping.cov = tr_state.filtered().covariance();
	}
    };

private:
    propagator_t m_propagator;



    template <typename parameters_t>
    class aborter {};
};

}  // namespace traccc
