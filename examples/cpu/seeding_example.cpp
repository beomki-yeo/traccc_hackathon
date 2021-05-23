/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "edm/internal_spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"

// seeding
#include "algorithms/seeding/spacepoint_grouping.hpp"
#include "algorithms/seeding/seed_finding.hpp"

// io
#include "csv/csv_io.hpp"

// vecmem
#include <vecmem/memory/host_memory_resource.hpp>

int seq_run(const std::string& detector_file, const std::string& hits_dir, unsigned int events)
{
    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d) + std::string("/");

    // Read the surface transforms
    std::string io_detector_file = data_directory + detector_file;
    traccc::surface_reader sreader(io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    auto surface_transforms = traccc::read_surfaces(sreader);

    // Output stats
    uint64_t n_spacepoints = 0;
    uint64_t n_internal_spacepoints = 0;
    uint64_t n_doublets = 0;
    uint64_t n_modules = 0;
    
    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Loop over events
    for (unsigned int event = 0; event < events; ++event){

        // Read the cells from the relevant event file
        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size()-event_number.size(), event_number.size(), event_number);

        std::string io_hits_file = data_directory+hits_dir+std::string("/event")+event_string+std::string("-hits.csv");
	
        traccc::fatras_hit_reader hreader(io_hits_file, {"particle_id","geometry_id","tx","ty","tz","tt","tpx","tpy","tpz","te","deltapx","deltapy","deltapz","deltae","index"});
	traccc::host_spacepoint_container spacepoints_per_event = traccc::read_hits(hreader, resource);

	
	for (size_t i=0; i<spacepoints_per_event.headers.size(); i++){
	    auto& spacepoints_per_module = spacepoints_per_event.items[i];
	    
	    n_spacepoints += spacepoints_per_module.size();
	    n_modules++;
	}

	/*-------------------
	     Seed finding
	  -------------------*/

	// Seed finder config
	traccc::seedfinder_config config;
	// silicon detector max
	config.rMax = 160.;
	config.deltaRMin = 5.;
	config.deltaRMax = 160.;	
	config.collisionRegionMin = -250.;
	config.collisionRegionMax = 250.;
	config.zMin = -2800.;
	config.zMax = 2800.;
	config.maxSeedsPerSpM = 5;
	// 2.7 eta
	config.cotThetaMax = 7.40627;
	config.sigmaScattering = 1.00000;
	
	config.minPt = 500.;
	config.bFieldInZ = 0.00199724;
	
	config.beamPos = {-.5, -.5};
	config.impactMax = 10.;

	// setup spacepoint grid config
	traccc::spacepoint_grid_config grid_config;
	grid_config.bFieldInZ = config.bFieldInZ;
	grid_config.minPt = config.minPt;
	grid_config.rMax = config.rMax;
	grid_config.zMax = config.zMax;
	grid_config.zMin = config.zMin;
	grid_config.deltaRMax = config.deltaRMax;
	grid_config.cotThetaMax = config.cotThetaMax;

	// create internal spacepoints grouped in bins
	traccc::spacepoint_grouping sg(config, grid_config);
	auto internal_sp_per_event = sg(spacepoints_per_event);

	// seed finding
	traccc::seed_finding sf(config, internal_sp_per_event);
	auto seeds = sf();
	
	for (size_t i=0; i<internal_sp_per_event.headers.size(); ++i){
	    n_internal_spacepoints+=internal_sp_per_event.items[i].size();
	}

	
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " <<  n_spacepoints << " spacepoints from " << n_modules << " modules" << std::endl;
    std::cout << "- created " <<  n_internal_spacepoints << " internal spacepoints" << std::endl;
    
    return 0;
}

// The main routine
//
int main(int argc, char *argv[])
{
    if (argc < 4){
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <hit_directory> <events>" << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto hit_directory = std::string(argv[2]);
    auto events = std::atoi(argv[3]);

    std::cout << "Running ./seeding_example " << detector_file << " " << hit_directory << " " << events << std::endl;
    return seq_run(detector_file, hit_directory, events);
}