#include <gtest/gtest.h>
#include <vecmem/memory/host_memory_resource.hpp>

#include "csv/csv_io.hpp"
#include <edm/measurement.hpp>
#include <edm/track_parameters.hpp>
#include <edm/track_state.hpp>
#include <edm/truth/truth_bound_track_parameters.hpp>
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

std::vector<std::pair<traccc::host_truth_bound_track_parameters_collection, traccc::host_measurement_collection>>
read_particles(traccc::host_surface_collection& surfaces)
{
	std::vector<std::pair<traccc::host_truth_bound_track_parameters_collection, traccc::host_measurement_collection>>
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
		traccc::host_truth_bound_track_parameters_collection tbps;
		 traccc::host_measurement_collection meas;
		 for (traccc::spacepoint& sp : spacepoints_per_event.items[i]) {
			 tbps.items.push_back(sp.make_bound_track_parameters(surfaces, t_particle));
			 meas.push_back(sp.make_measurement(surfaces));
			 meas.back().variance[0] = std::pow(0.001 * meas.back().local[0], 2);
			 meas.back().variance[1] = std::pow(0.001 * meas.back().local[1], 2);
		 }
		 retv.push_back(std::make_pair(std::move(tbps), std::move(meas)));
	}
	return retv;
}

traccc::host_surface_collection SURFACES = read_surfaces();
std::vector<std::pair<traccc::host_truth_bound_track_parameters_collection, traccc::host_measurement_collection>>
		TRUTH_DATA = read_particles(SURFACES);


TEST(algorithm, actor)
{
	traccc::bound_track_parameters bpars = TRUTH_DATA.at(0).first.items.at(0);
	bpars.covariance() = traccc::bound_track_parameters::covariance_t::Identity();
	traccc::host_measurement_collection  meas = {TRUTH_DATA.at(0).second.at(0)};

	stepper_t stepper;
	propagator_state_t prop_state;
	prop_state.stepping = stepper.make_state(bpars, SURFACES);
	actor_t actor(meas.at(0).surface_id, meas, SURFACES);

	Acts::FreeVector tpars_before = prop_state.stepping.pars;
	actor(prop_state, stepper);
	Acts::FreeVector tpars_after = prop_state.stepping.pars;

	// Re-update the parameters using the truth measurement at the
	// same surface, therefore we expect no change
	bool no_diff = true;
	bool has_nan = false;
	for (int i = 0; i < Acts::eFreeSize; i++) {
		std::cout << "#" << i << ": before=" << tpars_before[i] << " after=" << tpars_after[i] << std::endl;
		no_diff &= (tpars_before[i] - tpars_after[i]) < 1e-8;
		has_nan |= std::isnan(tpars_after[i]);
	}
	EXPECT_TRUE(no_diff);	
	EXPECT_FALSE(has_nan);	
}

TEST(algorithm, actor2)
{

	traccc::bound_track_parameters bpars = TRUTH_DATA.at(0).first.items.at(0);
	bpars.covariance() = traccc::bound_track_parameters::covariance_t::Identity();
	traccc::host_measurement_collection  meas = {TRUTH_DATA.at(0).second.at(0)};
	meas.at(0).local[0] *= 2;
	meas.at(0).local[1] *= 2;

	stepper_t stepper;
	propagator_state_t prop_state;
	prop_state.stepping = stepper.make_state(bpars, SURFACES);
	actor_t actor(meas.at(0).surface_id, meas, SURFACES);

	Acts::FreeVector tpars_before = prop_state.stepping.pars;
	actor(prop_state, stepper);
	Acts::FreeVector tpars_after = prop_state.stepping.pars;

	// Here the measurement was fudged so we expecte a change
	bool no_diff = true;
	bool has_nan = false;
	for (int i = 0; i < Acts::eFreeSize; i++) {
		std::cout << "#" << i << ": before=" << tpars_before[i] << " after=" << tpars_after[i] << std::endl;
		no_diff &= (tpars_before[i] - tpars_after[i]) < 1e-8;
		has_nan |= std::isnan(tpars_after[i]);
	}
	EXPECT_FALSE(no_diff);	
	EXPECT_FALSE(has_nan);	
}
