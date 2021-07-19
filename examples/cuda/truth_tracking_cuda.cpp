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
#include "geometry/pixel_segmentation.hpp"

// clusterization (cpu)
#include "clusterization/component_connection.hpp"
#include "clusterization/measurement_creation.hpp"
#include "clusterization/spacepoint_formation.hpp"

// seeding (cpu)
#include "seeding/seed_finding.hpp"
#include "seeding/spacepoint_grouping.hpp"
// seeding (cuda)
#include "cuda/seeding/seed_finding.hpp"

// track parmeter estimiation (cpu)
#include "seeding/track_params_estimating.hpp"

// fitter
#include "fitter/kalman_fitter.hpp"

// geometry
#include "geometry/surface.hpp"

// io
#include "csv/csv_io.hpp"

// gpuKalmanFilter
#include "Geometry/GeometryContext.hpp"
#include "MagneticField/MagneticFieldContext.hpp"
#include "Material/HomogeneousSurfaceMaterial.hpp"
#include "Test/Helper.hpp"
#include "Test/Logger.hpp"

// vecmem
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// std
#include <chrono>
#include <iomanip>

// custom
#include "tml_stats_config.hpp"

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

    using surface_type = Acts::PlaneSurface<Acts::InfiniteBounds>;
    
    traccc::host_surface_collection<surface_type> surface_vector(
        {traccc::host_surface_collection<surface_type>::item_vector(0,&mng_mr)});

    
    Acts::MaterialSlab matProp(Test::makeSilicon(), 0.5 * Acts::units::_mm);
    Acts::HomogeneousSurfaceMaterial surfaceMaterial(matProp);

    std::function<surface_type(traccc::transform3)>trans2surface = [&](traccc::transform3 trans)
    {
	auto normal = traccc::getter::block<3,1>(trans._data,0,2);
	auto center = traccc::getter::block<3,1>(trans._data,0,3);
       	
	Acts::Vector3D e_normal;
	e_normal[0] = normal[0][0];
	e_normal[1] = normal[1][0];
	e_normal[2] = normal[2][0];
	
	Acts::Vector3D e_center;
	e_center[0] = center[0][0];
	e_center[1] = center[1][0];
	e_center[2] = center[2][0];
	
	auto surface = surface_type(e_center, e_normal, surfaceMaterial);
	return surface;
    };

    // Fill surface_collection
    for (auto trans: surface_transforms){
	auto geometry = trans.first;
	surface_type surface = trans2surface(trans.second);
	traccc::surface_link<surface_type> s_link({geometry, surface});
	surface_vector.items.push_back(std::move(s_link));	
    }
    
    // Context
    Acts::GeometryContext gctx(0);
    Acts::MagneticFieldContext mctx(0);
    
    // Algorithms
    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
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

        traccc::fatras_hit_reader hreader(
            io_hits_file,
            {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
             "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});
        traccc::host_spacepoint_container spacepoints_per_event =
            traccc::read_hits(hreader, resource);

        for (size_t i = 0; i < spacepoints_per_event.headers.size(); i++) {
            auto& spacepoints_per_module = spacepoints_per_event.items[i];

            n_spacepoints += spacepoints_per_module.size();
            n_modules++;
        }	
	
        /*time*/ auto end_file_reading_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_file_reading_cpu =
            end_file_reading_cpu - start_file_reading_cpu;
        /*time*/ file_reading_cpu += time_file_reading_cpu.count();	
	

        /*---------------------------------------------------
             Local Transformation (spacepoint->measurement)
          ---------------------------------------------------*/

        traccc::host_measurement_container measurements_per_event({
            traccc::host_measurement_container::header_vector(0, &mng_mr),
	    traccc::host_measurement_container::item_vector(0,&mng_mr)});	
	
	for (unsigned int i_h=0; i_h<spacepoints_per_event.headers.size(); i_h++){
	    // fill measurement headers
	    const auto& geometry = spacepoints_per_event.headers[i_h];
	    auto placement = surface_transforms[geometry];

	    surface_type surface = trans2surface(placement);
	    traccc::surface_link<surface_type> s_link({geometry, surface});
	    
	    auto it = std::find_if(surface_vector.items.begin(),
				   surface_vector.items.end(),
				   [&s_link](auto& tmp_link){
				       return s_link.geometry == tmp_link.geometry;
				   });
	    auto surface_id = std::distance(surface_vector.items.begin(), it);

	    traccc::cell_module module({event, geometry, surface_id, placement});
	    measurements_per_event.headers.push_back(module);
	    
	    // fill measurement items	    
	    const auto& spacepoints_per_module = spacepoints_per_event.items[i_h];
	    traccc::host_measurement_collection measurements_per_module;

	    for (auto sp: spacepoints_per_module){
		const auto& pos = sp.global_position();
		auto loc3 = placement.point_to_local(pos);
		// Note: loc3[2] should be equal or very close to 0
		traccc::point2 loc({loc3[0],loc3[1]});
		// Todo: smear the loc (What is a good value for variance?)
		traccc::variance2 var({0,0}); 
		traccc::measurement ms({loc, var, i_h, surface_id, sp.pid});

		measurements_per_module.push_back(ms);
	    }

	    measurements_per_event.items.push_back(measurements_per_module);
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
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created        " << n_cells
              << " cells           " << std::endl;
    std::cout << "- created        " << n_clusters
              << " clusters        " << std::endl;        
    std::cout << "- created        " << n_measurements
              << " meaurements     " << std::endl;
    std::cout << "- created        " << n_spacepoints
              << " spacepoints     " << std::endl;
    std::cout << "- created        " << n_internal_spacepoints
              << " internal spacepoints" << std::endl;

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
