/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <iostream>
#include <vecmem/memory/host_memory_resource.hpp>

#include "csv/csv_io.hpp"
#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/internal_spacepoint.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "edm/truth/truth_measurement.hpp"
#include "edm/truth/truth_spacepoint.hpp"
#include "edm/truth/truth_bound_track_parameters.hpp"
#include "geometry/pixel_segmentation.hpp"
#include "clusterization/spacepoint_formation.hpp"

// fitter
#include "fitter/kalman_fitter.hpp"

// geometry
#include "geometry/surface.hpp"

// io
#include "csv/csv_io.hpp"

// vecmem
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// std
#include <chrono>
#include <iomanip>

// custom
#include "tml_stats_config.hpp"

#include "Acts/Utilities/Helpers.hpp"

int seq_run(const std::string& detector_file, const std::string& hits_dir,
	    unsigned int skip_events, unsigned int events, bool skip_cpu,
            bool skip_write) {

    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    
    // Read the surface transforms
    std::string io_detector_file = detector_file;
    traccc::surface_reader sreader(
        io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv",
                           "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    auto surface_transforms = traccc::read_surfaces(sreader);
    std::map<traccc::geometry_id, uint64_t> surface_link;
    
    /*---------------------
      surface collection
     ---------------------*/

    // surface collection where all surfaces are saved
    traccc::host_surface_collection surfaces(
        {traccc::host_surface_collection::item_vector(0,&mng_mr)});

    // Let me ignore material property for the moment...    
    //Acts::MaterialSlab matProp(Test::makeSilicon(), 0.5 * Acts::units::_mm);
    //Acts::HomogeneousSurfaceMaterial surfaceMaterial(matProp);

    // Fill surface_collection
    for (auto tf: surface_transforms){
	traccc::surface surface(tf.second, tf.first);
	surfaces.items.push_back(std::move(surface));	
    }

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_particles = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_internal_spacepoints = 0;
    uint64_t n_doublets = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;

    // Elapsed time
    float wall_time(0);

    float file_reading_cpu(0);
    float clusterization_cpu(0);	
    float measurement_creation_cpu(0);
    float spacepoint_formation_cpu(0);
    float binning_cpu(0);
    float seeding_cpu(0);
    float tp_estimation_cpu(0);

    float clusterization_cuda(0);	
    float measurement_creation_cuda(0);
    float spacepoint_formation_cuda(0);
    float binning_cuda(0);
    float seeding_cuda(0);   
    
    /*time*/ auto start_wall_time = std::chrono::system_clock::now();
    
    // Loop over events
    for (unsigned int event = skip_events; event < skip_events + events;
         ++event) {

	/*time*/ auto start_file_reading_cpu = std::chrono::system_clock::now();
	
        // Read the hits from the relevant event file       
        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size() - event_number.size(),
                             event_number.size(), event_number);

        std::string io_hits_file = hits_dir + std::string("/event") +
                                   event_string + std::string("-hits.csv");

        std::string io_particle_file = hits_dir + std::string("/event") +
	    event_string + std::string("-particles_initial.csv");

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
	    traccc::read_truth_hits(hreader, preader, resource);
	
        /*time*/ auto end_file_reading_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_file_reading_cpu =
            end_file_reading_cpu - start_file_reading_cpu;
        /*time*/ file_reading_cpu += time_file_reading_cpu.count();	
	

        /*---------------------------------------------------
	  Global to Local Transformation (spacepoint->measurement)
          ---------------------------------------------------*/

	n_particles = spacepoints_per_event.headers.size();

	// declare truth measurement container
	traccc::host_truth_measurement_container measurements_per_event({
            traccc::host_truth_measurement_container::header_vector(n_particles, &mng_mr),
	    traccc::host_truth_measurement_container::item_vector(n_particles,&mng_mr)});

	// declare truth bound parameters container
	traccc::host_truth_bound_track_parameters_container bound_track_parameters_per_event({
            traccc::host_truth_bound_track_parameters_container::header_vector(n_particles, &mng_mr),
	    traccc::host_truth_bound_track_parameters_container::item_vector(n_particles,&mng_mr)});
	
	// fill measurement container
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

	    // count spacepoints/measurements
            n_spacepoints += spacepoints_per_particle.size();
	    n_measurements += measurements_per_particle.size();
	}

	/*----------------------------------------------------
	  Start Truth Tracking Here
	  ---------------------------------------------------*/

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
	         	
	// iterate over truth particles
	for (int i_h = 0; i_h < measurements_per_event.headers.size(); i_h++){
	    auto& t_particle = measurements_per_event.headers[i_h];
	    auto& measurements_per_particle = measurements_per_event.items[i_h];
	    auto& bound_track_parameters_per_particle = bound_track_parameters_per_event.items[i_h];
	    
	    // Do the tracking here
	    
	}
	
        /*------------
             Writer
          ------------*/

	if (!skip_write){
	    // leave empty for the moment
	}
    }

    /*time*/ auto end_wall_time = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> time_wall_time =
        end_wall_time - start_wall_time;

    /*time*/ wall_time += time_wall_time.count();

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- created        " << n_particles
              << " particles       " << std::endl;
    std::cout << "- created        " << n_measurements
              << " meaurements     " << std::endl;
    std::cout << "- created        " << n_spacepoints
              << " spacepoints     " << std::endl;

    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cuda) " << n_seeds_cuda << " seeds" << std::endl;
    std::cout << "==> Elpased time ... " << std::endl;
    std::cout << "wall time           " << std::setw(10) << std::left
              << wall_time << std::endl;
    std::cout << "file reading (cpu)        " << std::setw(10) << std::left
              << file_reading_cpu << std::endl;
    std::cout << "clusterization_time (cpu) " << std::setw(10) << std::left
              << clusterization_cpu << std::endl;    
    std::cout << "ms_creation_time (cpu)    " << std::setw(10) << std::left
              << measurement_creation_cpu << std::endl;    
    std::cout << "sp_formation_time (cpu)   " << std::setw(10) << std::left
              << spacepoint_formation_cpu << std::endl;    
    std::cout << "binning_time (cpu)        " << std::setw(10) << std::left
              << binning_cpu << std::endl;
    std::cout << "seeding_time (cpu)        " << std::setw(10) << std::left
              << seeding_cpu << std::endl;
    std::cout << "seeding_time (cuda)       " << std::setw(10) << std::left
              << seeding_cuda << std::endl;
    std::cout << "tp_estimation_time (cuda) " << std::setw(10) << std::left
              << tp_estimation_cpu << std::endl;

    
    return 0;    
}

// The main routine
//
int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <hit_directory> "
                     "<skip_events> <events> <skip_cpu> <skip_write>"
                  << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto hit_directory = std::string(argv[2]);
    auto skip_events = std::atoi(argv[3]);
    auto events = std::atoi(argv[4]);
    bool skip_cpu = std::atoi(argv[5]);
    bool skip_write = std::atoi(argv[6]);

    std::cout << "Running ./seq_example " << detector_file << " "
              << hit_directory << " " << skip_events << " " << events << " "
              << skip_cpu << " " << skip_write << std::endl;
    return seq_run(detector_file, hit_directory, skip_events, events, skip_cpu,
                   skip_write);
}
