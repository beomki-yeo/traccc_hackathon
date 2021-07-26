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

#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// traccc core
#include <edm/measurement.hpp>
#include <edm/track_parameters.hpp>
#include <edm/track_state.hpp>

// std
#include <chrono>

// This defines the local frame test suite
TEST(algebra, convariance_engine) {}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
