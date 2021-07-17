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
#include <fitter/gain_matrix_updater.hpp>
#include <cuda/fitter/gain_matrix_updater.cuh>

// std
#include <chrono>

// This defines the local frame test suite
TEST(algebra, gain_matrix_updater) {    
    const int batch_size = 5000;
    const int event_size = 1000;

    // The managed memory resource
    vecmem::cuda::managed_memory_resource mng_mr;

    // The host/device memory resources
    vecmem::cuda::device_memory_resource dev_mr;
    vecmem::cuda::host_memory_resource host_mr;

    // memory copy helper
    vecmem::cuda::copy m_copy;
    
    // define track_state type
    using scalar_t = double;
    using measurement_t = traccc::measurement;
    using bound_track_parameters_t = traccc::bound_track_parameters;
    
    using track_state_t = traccc::track_state<measurement_t, bound_track_parameters_t>;
    using host_track_state_collection =
	traccc::host_track_state_collection<track_state_t>;

    // declare a test track states object
    host_track_state_collection track_states_cpu({host_track_state_collection::item_vector(&mng_mr)});    
    host_track_state_collection track_states_gpu({host_track_state_collection::item_vector(&mng_mr)});
    
    // fillout random elements to track states
    for (int i_b=0; i_b < batch_size; i_b++){
	track_state_t tr_state;

	tr_state.predicted().vector() = bound_track_parameters_t::vector_t::Random();
	tr_state.predicted().covariance() = bound_track_parameters_t::covariance_t::Random();
	
	tr_state.filtered().vector() = bound_track_parameters_t::vector_t::Random();
	tr_state.filtered().covariance() = bound_track_parameters_t::covariance_t::Random();
	
	tr_state.smoothed().vector() = bound_track_parameters_t::vector_t::Random();
	tr_state.smoothed().covariance() = bound_track_parameters_t::covariance_t::Random();

	tr_state.jacobian() = track_state_t::jacobian_t::Random();
	tr_state.projector() = track_state_t::projector_t::Identity();

	track_states_cpu.items.push_back(tr_state);
	track_states_gpu.items.push_back(tr_state);
    }

    // for timing benchmark
    float time_cpu(0);
    float time_gpu(0);
   
    // cpu gain matrix updater

    traccc::gain_matrix_updater<track_state_t> cpu_updater;

    /*time*/ auto start_cpu = std::chrono::system_clock::now();

    for (int i_n = 0; i_n < event_size; i_n++){
    
	for (auto& tr_state: track_states_cpu.items){
	    cpu_updater(tr_state);
	}

    }
    
    /*time*/ auto end_cpu = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> elapse_cpu = end_cpu - start_cpu;

    /*time*/ time_cpu += elapse_cpu.count();
    
    // cuda gain matrix updater

    traccc::cuda::gain_matrix_updater<track_state_t> cuda_updater;

    /*time*/ auto start_gpu = std::chrono::system_clock::now();

    for (int i_n = 0; i_n < event_size; i_n++){
	
	cuda_updater(track_states_gpu, &mng_mr);
    
    }
    
    /*time*/ auto end_gpu = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> elapse_gpu = end_gpu - start_gpu;

    /*time*/ time_gpu += elapse_gpu.count();
    
    // check if the result is the same
    for (int i_t = 0; i_t < track_states_cpu.items.size(); i_t++){
	auto& tr_state_cpu = track_states_cpu.items[i_t];
	auto& tr_state_gpu = track_states_gpu.items[i_t];

	auto& cpu_vec = tr_state_cpu.filtered().vector();
	auto& cpu_cov = tr_state_cpu.filtered().covariance();

	auto& gpu_vec = tr_state_gpu.filtered().vector();
	auto& gpu_cov = tr_state_gpu.filtered().covariance();

	
	for( std::size_t i = 0; i < cpu_vec.rows()*cpu_vec.cols(); ++i ) {
	    EXPECT_TRUE( abs(cpu_vec(i) -gpu_vec(i)) < 1e-8 );
	}

	for( std::size_t i = 0; i < cpu_cov.rows()*cpu_cov.cols(); ++i ) {
	    EXPECT_TRUE( abs(cpu_cov(i) -gpu_cov(i)) < 1e-8 );
	}
		
	//std::cout << tr_state_cpu.filtered().vector() << std::endl;
	//std::cout << tr_state_gpu.filtered().vector() << std::endl;
	
	//std::cout << tr_state_cpu.filtered().covariance() << std::endl;
	//std::cout << tr_state_gpu.filtered().covariance() << std::endl;	
    }

    std::cout << "==> Elpased time ... " << std::endl;
    std::cout << "cpu time: " << time_cpu << std::endl;
    std::cout << "gpu time: " << time_gpu << std::endl;
    
}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
