#include <gtest/gtest.h>
#include <vecmem/memory/host_memory_resource.hpp>

#include "csv/csv_io.hpp"
#include <edm/measurement.hpp>
#include <edm/track_parameters.hpp>
#include <edm/track_state.hpp>
#include <fitter/gain_matrix_smoother.hpp>
#include <fitter/gain_matrix_updater.hpp>
#include <fitter/kalman_fitter.hpp>
#include "geometry/surface.hpp"
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
using parameters_t = traccc::bound_track_parameters;
using track_state_t = traccc::track_state<measurement_t, parameters_t>;
using updater_t = traccc::gain_matrix_updater<track_state_t>;
using smoother_t = traccc::gain_matrix_smoother<track_state_t>;
using kalman_fitter_t = traccc::kalman_fitter<propagator_t, updater_t, smoother_t>;
using actor_t = kalman_fitter_t::actor<parameters_t>;


traccc::host_surface_collection read_surfaces()
{
	std::string full_name = std::string(__FILE__);
	std::string dir = full_name.substr(0, full_name.find_last_of("\\/"));
	std::string io_detector_file =
		dir + std::string("/detector/trackml-detector.csv");

	traccc::surface_reader sreader(
		io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv",
					      "rot_xw", "rot_zu", "rot_zv", "rot_zw"});

	auto surface_transforms = traccc::read_surfaces(sreader);

	traccc::host_surface_collection surfaces;

	for (auto tf : surface_transforms) {
		traccc::surface surface(tf.second, tf.first);
		surfaces.items.push_back(std::move(surface));
	}

	return surfaces;
}

std::vector<std::pair<traccc::free_track_parameters, traccc::host_measurement_collection>>
read_particles(traccc::host_surface_collection& surfaces)
{
	// Need: *local* measurements
	// Need: *global* track parameters
	std::vector<std::pair<traccc::free_track_parameters, traccc::host_measurement_collection>>
		retv;

	std::string full_name = std::string(__FILE__);
	std::string dir = full_name.substr(0, full_name.find_last_of("\\/"));

	std::string io_hits_file = dir + std::string("/data/hits.csv");
	std::string io_particle_file = dir + std::string("/data/particles.csv");

	traccc::fatras_hit_reader hreader(
		io_hits_file,
		{"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
		 "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});

	// truth particle reader
	traccc::fatras_particle_reader preader(
		io_particle_file,
		{"particle_id", "particle_type", "vx", "vy", "vz",
		 "vt", "px", "py", "pz", "m", "q"});

	// read truth hits
	vecmem::host_memory_resource host_mr;
	traccc::host_truth_spacepoint_container spacepoints_per_event =
		traccc::read_truth_hits(hreader, preader, host_mr);

	int n_particles = spacepoints_per_event.headers.size();

	for (size_t i = 0; i < n_particles; i++) {
		traccc::truth_particle& t_particle = spacepoints_per_event.headers[i];
		 traccc::host_measurement_collection meas;
		 // std::cout << t_particle.vertex.vector()[0] << " " <<  t_particle.vertex.vector()[1] << " " <<  t_particle.vertex.vector()[2] << std::endl;
		 // std::cout << "-----" << std::endl;
		 for (traccc::spacepoint& sp : spacepoints_per_event.items[i]) {
			 meas.push_back(sp.make_measurement(surfaces));
			 if (i == 0) {
				 std::cout << meas.back().local[0] << " " << meas.back().local[1] <<  " " << meas.back().variance[0] << " " << meas.back().variance[1] << " " << meas.back().surface_id << std::endl;
			 }
			 meas.back().variance[0] = 0.001 * meas.back().local[0];
			 meas.back().variance[1] = 0.001 * meas.back().local[1];
		 }
		 retv.push_back(std::make_pair(t_particle.vertex, std::move(meas)));
	}
	return retv;
}


TEST(algorithm, actor)
{
	traccc::host_surface_collection surfaces = read_surfaces();

	measurement_t meas;
	meas.local = {1, 1};
	meas.variance = {1, 1};
	meas.surface_id = 0;

	traccc::host_measurement_collection meascoll = { meas };

	stepper_t stepper;
	propagator_state_t prop_state;

	actor_t actor(meas.surface_id, meascoll, surfaces);

	Acts::FreeVector tpars_before = prop_state.stepping.pars;
	actor(prop_state, stepper);
	Acts::FreeVector tpars_after = prop_state.stepping.pars;

	bool any_diff = false;
	bool has_nan = false;
	for (int i = 0; i < Acts::eFreeSize; i++) {
		any_diff |= (tpars_before[i] != tpars_after[i]);
		has_nan |= std::isnan(tpars_after[i]);
	}

	EXPECT_TRUE(any_diff);	
	EXPECT_FALSE(has_nan);	
}

void dump(stepper_t::state &state)
{
	for (size_t i = 0; i < Acts::eFreeSize; i++) {
		std::cout << state.pars[i] << " ";
	}
	std::cout << "\n---------------------" << std::endl;
		
}

std::vector<int> get_surfaces_for_this_track(traccc::host_measurement_collection &coll)
{
	std::vector<int> retv;
	for (traccc::measurement & meas : coll) {
		retv.push_back(meas.surface_id);
	}
	return retv;
}

TEST(aglorithm, actor)
{
	traccc::host_surface_collection surfaces = read_surfaces();
	std::vector<std::pair<traccc::free_track_parameters, traccc::host_measurement_collection>>
		truth_data = read_particles(surfaces);

	
	// FIXME: This test is bad
	// Should just test piece-wise: (Truth State, Measurement) -> Truth State and check it has not changed
	// Then (TruthState + Perturbation, Measurment) and check that it messes it up
	// Then (TruthState, Measurement + Perturbation)
	// Could also compute by hand the result for a very simple output
	for (auto& [t_pars, meas] : truth_data) {
		stepper_t stepper;
		propagator_state_t prop_state;
		prop_state.stepping.nav_dir = Acts::forward;
		prop_state.stepping.q = (t_pars.qop() >= 0)? 1 : -1;
		prop_state.stepping.pars = t_pars.vector();

		prop_state.stepping.cov = 0.001*Acts::BoundSymMatrix::Random();//t_pars.covariance();
		dump(prop_state.stepping);
		for (int i : get_surfaces_for_this_track(meas)) {
			actor_t actor(i, meas, surfaces);
			actor(prop_state, stepper);
			dump(prop_state.stepping);
		}
		dump(prop_state.stepping);

		// FIXME better check
		EXPECT_TRUE((t_pars.qop() - prop_state.stepping.pars[Acts::eFreeQOverP]) < 1e-6);
		break;
	}
}
