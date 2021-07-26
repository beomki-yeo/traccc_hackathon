#include <gtest/gtest.h>

#include <edm/measurement.hpp>
#include <edm/track_parameters.hpp>
#include <edm/track_state.hpp>
#include <fitter/gain_matrix_smoother.hpp>
#include <fitter/gain_matrix_updater.hpp>
#include <fitter/kalman_fitter.hpp>
#include <propagator/direct_navigator.hpp>
#include <propagator/eigen_stepper.hpp>
#include <propagator/propagator.hpp>
#include <propagator/propagator_options.hpp>

using stepper_t = traccc::eigen_stepper;
using navigator_t = traccc::direct_navigator;
using propagator_t = traccc::propagator<stepper_t, navigator_t>;
using propagator_options_t = traccc::propagator_options<traccc::void_actor, traccc::void_aborter>;
using propagator_state_t = propagator_t::state<propagator_options_t>;
using measurement_t = traccc::measurement;
using parameters_t = traccc::free_track_parameters;
using track_state_t = traccc::track_state<measurement_t, parameters_t>;
using updater_t = traccc::gain_matrix_updater<track_state_t>;
using smoother_t = traccc::gain_matrix_smoother<track_state_t>;
using kalman_fitter_t = traccc::kalman_fitter<propagator_t, updater_t, smoother_t>;
using actor_t = kalman_fitter_t::actor<parameters_t>;

#include <edm/measurement.hpp>

TEST(algorithm, actor)
{
	measurement_t meas;
	meas.local = {1, 1};
	meas.variance = {1, 1};
	meas.surface_id = 1;

	traccc::host_measurement_collection meascoll = { meas };

	stepper_t stepper;
	propagator_state_t prop_state;

	actor_t actor;
	actor.target_surface_id = prop_state.navigation.current_surface_id;
	actor.input_measurements = &meascoll;

	Acts::FreeVector tpars_before = prop_state.stepping.pars;
	actor(prop_state, stepper);
	Acts::FreeVector tpars_after = prop_state.stepping.pars;

	bool any_diff = false;
	for (int i = 0; i < Acts::eFreeSize; i++)
		any_diff |= (tpars_before != tpars_after);

	EXPECT_TRUE(any_diff);	
}
