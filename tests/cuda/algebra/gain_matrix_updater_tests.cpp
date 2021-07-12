/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

// vecmem
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// traccc core
#include <edm/measurement.hpp>
#include <edm/track_parameters.hpp>
#include <edm/track_state.hpp>
#include <cuda/fitter/gain_matrix_updater.hpp>


// This defines the local frame test suite
TEST(algebra, gain_matrix_updater) {    
    const int batch_size = 1000;

    vecmem::cuda::managed_memory_resource mng_mr;

    // define track_state type
    using scalar_t = double;
    using measurement_t = traccc::measurement;
    using bound_track_parameters_t = traccc::bound_track_parameters;
    
    using track_state = traccc::track_state<measurement_t, bound_track_parameters_t>;
    using host_track_state_collection =
	traccc::host_track_state_collection<measurement_t, bound_track_parameters_t>;
    
    // declare a test track states object
    host_track_state_collection track_states({host_track_state_collection::item_vector(&mng_mr)});
    
    // fillout random elements to track states
    for (int i_b=0; i_b < batch_size; i_b++){
	track_state tr_state;

	tr_state.predicted().vector() = bound_track_parameters_t::vector_t::Random();
	tr_state.predicted().covariance() = bound_track_parameters_t::covariance_t::Random();

	tr_state.filtered().vector() = bound_track_parameters_t::vector_t::Random();
	tr_state.filtered().covariance() = bound_track_parameters_t::covariance_t::Random();
	
	tr_state.smoothed().vector() = bound_track_parameters_t::vector_t::Random();
	tr_state.smoothed().covariance() = bound_track_parameters_t::covariance_t::Random();

	tr_state.jacobian() = track_state::jacobian_t::Random();

	tr_state.projector() = track_state::projector_t::Identity();

	tr_state.projector2() = track_state::projector_t::Zero();
	
	// Todo: fill measurements
	
	track_states.items.push_back(tr_state);
    }

    // Declare gain matrix updater
    traccc::cuda::gain_matrix_updater<scalar_t, 2, Acts::eBoundSize, batch_size> cuUpdater;
    // Update the predicted -> filtered
    cuUpdater.update(track_states);
    
    /*
    traccc::host_bound_track_parameters_collection track_states(
								{traccc::host_bound_track_parameters_collection::item_vector(&mng_mr)});

    // fillout random elements to track states
    for (int i_b=0; i_b < batch_size; i_b++){
	traccc::bound_track_parameters tr_param;
	tr_param.params() = Acts::BoundVector::Random();
	tr_param.cov() = Acts::BoundSymMatrix::Random();

	track_states.items.push_back(std::move(tr_param));
    }
           
    traccc::cuda::gain_matrix_updater<double, matrix_dim, batch_size> cuUpdater;

    cuUpdater.update(track_states);        
    */
}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
