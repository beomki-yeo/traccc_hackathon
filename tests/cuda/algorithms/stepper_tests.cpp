/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

// vecmem
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// traccc core
#include <edm/measurement.hpp>
#include <edm/track_parameters.hpp>
#include <edm/track_state.hpp>
#include <propagator/propagator.hpp>
#include <propagator/propagator_options.hpp>
#include <propagator/detail/void_propagator_options.hpp>
#include <propagator/eigen_stepper.hpp>
#include <propagator/direct_navigator.hpp>
#include "geometry/surface.hpp"
#include "edm/truth/truth_measurement.hpp"
#include "edm/truth/truth_spacepoint.hpp"
#include "edm/truth/truth_bound_track_parameters.hpp"

// traccc cuda
#include <cuda/propagator/eigen_stepper.cuh>
#include <cuda/propagator/direct_navigator.hpp>
#include <cuda/propagator/propagator.hpp>

// std
#include <chrono>
#include <unistd.h>

// io
#include "csv/csv_io.hpp"


// This defines the local frame test suite
TEST(algebra, stepper) {
    
    /*-------------------
      Surface Reading
      -------------------*/
    
    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    
    // Geometry file
    std::string full_name = std::string(__FILE__);
    std::string dir = full_name.substr(0, full_name.find_last_of("\\/"));
    std::string io_detector_file = dir + std::string("/detector/trackml-detector.csv");

    // surface reader
    traccc::surface_reader sreader(
        io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv",
                           "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    // read surface
    auto surface_transforms = traccc::read_surfaces(sreader);
        
    // surface collection where all surfaces are saved
    traccc::host_surface_collection surfaces(
        {traccc::host_surface_collection::item_vector(0,&mng_mr)});

    // fill surface collection
    for (auto tf: surface_transforms){
	//std::cout << tf.first << std::endl;
	
	traccc::surface surface(tf.second, tf.first);

	//std::cout << surface.geom_id() << std::endl;
	
	surfaces.items.push_back(std::move(surface));	
    }

    /*-------------------
      Event Reading
      -------------------*/
    
    // Event file
    std::string io_hits_file = dir + std::string("/data/hits.csv");    
    std::string io_particle_file = dir + std::string("/data/particles.csv");

    // truth hit reader
    traccc::fatras_hit_reader hreader(
            io_hits_file,
            {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
             "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});

    // truth particle reader
    traccc::fatras_particle_reader preader(
            io_particle_file,
            {"particle_id", "particle_type", "vx", "vy", "vz", "vt", "px", "py",
             "pz", "m", "q"});

    // read truth hits
    traccc::host_truth_spacepoint_container spacepoints_per_event =
	traccc::read_truth_hits(hreader, preader, mng_mr);

    int n_particles = spacepoints_per_event.headers.size();
    
    // declare truth measurement container
    traccc::host_truth_measurement_container measurements_per_event({
       traccc::host_truth_measurement_container::header_vector(n_particles, &mng_mr),
       traccc::host_truth_measurement_container::item_vector(n_particles,&mng_mr)});

    // declare truth bound parameters container
    traccc::host_truth_bound_track_parameters_container bound_track_parameters_per_event({
       traccc::host_truth_bound_track_parameters_container::header_vector(n_particles, &mng_mr),
       traccc::host_truth_bound_track_parameters_container::item_vector(n_particles,&mng_mr)});

    // fill measurement and bound_track_parameter container
    for (unsigned int i_h=0; i_h<spacepoints_per_event.headers.size(); i_h++){
	auto& t_particle = spacepoints_per_event.headers[i_h];
	
	measurements_per_event.headers[i_h] = t_particle;
	bound_track_parameters_per_event.headers[i_h] = t_particle;
	
	auto& measurements_per_particle = measurements_per_event.items[i_h];
	auto& bound_track_parameters_per_particle = bound_track_parameters_per_event.items[i_h];
	
	auto& spacepoints_per_particle = spacepoints_per_event.items[i_h];
	
	for (auto sp: spacepoints_per_particle){
	    auto ms = sp.make_measurement(surfaces);
	    auto params = sp.make_bound_track_parameters(surfaces, t_particle);
	    
	    measurements_per_particle.push_back(ms);
	    bound_track_parameters_per_particle.push_back(params);		
	}	
    }    

    /*-------------------
      Do test from here
      -------------------*/

    /*----------------------------------------------------
      Note - important!
      
      You will use two vecmem "containers" as event data model
      : (1) measurements_per_event and (2) bound_track_parameters_per_event
      You will also use surface "collection" as geometry
      : (3) surfaces
      
      vecmem "container" consists of header and item,
      where header is vector and item is vector<vector>
      
      vecmem "collection" consists of item, where item is vector

      (1) measurement container 
      (1.A) header is the vector of truth particle which contains particle id, particle type, mass and vertex information
      (1.B) item is the vector of vector of measurement where each subvector is associated with each element of header. you can access surface by using surface_id member varaible

      ex) measurement ms; 
          surface reference_surf = surfaces.items[ms.surface_id];
      
      (2) bound_track_parameter container
      (2.A) header is the vector of truth particle which contains particle id, particle type, mass and vertex information -> same with measurement container
      (2.B) item is the vector of vector of bound_track_parameter consisting of vector (2-dim position, mom, time) and its covariance (+ surface_id) 
      
      ex) bound_track_parameters bp;
          Acts::BoundVector vec = bp.vector();
	  Acts::BoundSymMatrix cov = bp.covariance();
	  surface reference_surf = bp.reference_surface(surfaces);
      
      
      (3) surface collection: just vector of surfaces
      you can do global <-> local transformation with surface object
      
      ---------------------------------------------------*/
								     
    /*---------
      For CPU
      ---------*/

    // define tracking components
    using stepper_t = typename traccc::eigen_stepper;
    using stepper_state_t = typename traccc::eigen_stepper::state;
    using navigator_t = typename traccc::direct_navigator<traccc::surface>;
    using navigator_state_t = typename traccc::direct_navigator<traccc::surface>::state;
    using propagator_t = typename traccc::propagator<stepper_t, navigator_t>;
    using propagator_options_t = typename traccc::void_propagator_options;
    using propagator_state_t = typename propagator_t::state<propagator_options_t>;

    propagator_options_t void_po;
    
    // iterate over truth particles
    for (int i_h = 0; i_h < measurements_per_event.headers.size(); i_h++){

	stepper_t stepper;	
	navigator_t navigator;	
	propagator_t prop(stepper, navigator);       
	
	// truth particle information
	auto& t_particle = measurements_per_event.headers[i_h];

	// vector of spacepoints associated with a truth particle
	auto& spacepoints_per_particle = spacepoints_per_event.items[i_h];
	
	// vector of measurements associated with a truth particle
	auto& measurements_per_particle = measurements_per_event.items[i_h];

	// vector of bound_track_parameters associated with a truth particle
	auto& bound_track_parameters_per_particle = bound_track_parameters_per_event.items[i_h];
	
	// steper state
	stepper_state_t stepper_state(bound_track_parameters_per_particle[0],
				      surfaces);

	// propagator state that takes stepper state as input
	propagator_state_t prop_state(void_po, stepper_state);

	/*
	for (auto sp: spacepoints_per_particle){
	    std::cout << sp.x() << "  " << sp.y() << "  " << sp.z() << std::endl;
	}
	*/
	
	// fill the surface seqeunce
	auto& surf_seq = prop_state.navigation.surface_sequence;
	auto& surf_seq_size = prop_state.navigation.surface_sequence_size;       
	for (auto ms: measurements_per_particle){
	    surf_seq[surf_seq_size] = ms.surface_id;
	    surf_seq_size++;

	    if (surf_seq_size >= 30){
		std::cout << "too many surfaces!" << std::endl;
	    }
	}

	// manipulate eigen stepper state
	auto& sd = prop_state.stepping.step_data;

	// set B Field to 2T
	sd.B_first = Acts::Vector3(0,0,2*Acts::UnitConstants::T);
	sd.B_middle = Acts::Vector3(0,0,2*Acts::UnitConstants::T);
	sd.B_last = Acts::Vector3(0,0,2*Acts::UnitConstants::T);

	// do the eigen stepper
	for (int i_s = 0; i_s < prop_state.options.maxSteps ; i_s++){
	    auto navi_res = navigator_t::status(prop_state, surfaces);

	    if (!navi_res){
		std::cout << "Total RK steps: " << i_s << std::endl;
		std::cout << "all targets reached" << std::endl;
		break;
	    }
	    
	    auto& stepper_state = prop_state.stepping;
	    	    
	    auto res = stepper_t::rk4(prop_state);

	    //std::cout << stepper_state.step_size << " " << stepper_state.pars[Acts::eFreePos0] << "  " << stepper_state.pars[Acts::eFreePos1] << "  " << stepper_state.pars[Acts::eFreePos2] << std::endl;
	    
	    if (!res){
		std::cout << "stepping failed" << std::endl;
		break;
	    }

	    // do the covaraince transport
	    stepper_t::cov_transport(prop_state);
	}	
    }    

    /*---------
      For GPU
      ---------*/
    
    using cuda_stepper_t = traccc::cuda::eigen_stepper;
    using cuda_navigator_t = traccc::cuda::direct_navigator<traccc::surface>;;
    using cuda_propagator_t = traccc::cuda::propagator<cuda_stepper_t, cuda_navigator_t>;    
    using cuda_propagator_state_t = cuda_propagator_t::state<propagator_options_t>;
    
    // iterate over truth particles
    std::vector<traccc::bound_track_parameters> bp_collection;
    
    for (int i_h = 0; i_h < measurements_per_event.headers.size(); i_h++){

	auto& bound_track_parameters_per_particle
	    = bound_track_parameters_per_event.items[i_h];

	bp_collection.push_back(bound_track_parameters_per_particle[0]);	
    } 
    
    cuda_propagator_state_t cuda_prop_states(bp_collection, void_po, &mng_mr);
    // do the RK4
    auto res = cuda_stepper_t::rk4(cuda_prop_states);        
}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

