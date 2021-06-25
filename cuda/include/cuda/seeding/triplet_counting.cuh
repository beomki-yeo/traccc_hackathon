/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/seeding/detail/doublet_counter.hpp>
#include <cuda/seeding/detail/triplet_counter.hpp>
#include <cuda/utils/definitions.hpp>
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/triplet.hpp>
#include <seeding/doublet_finding_helper.hpp>
#include <seeding/triplet_finding_helper.hpp>

#pragma once

namespace traccc {
namespace cuda {

void triplet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      host_doublet_container& mid_bot_doublet_container,
                      host_doublet_container& mid_top_doublet_container,
                      host_triplet_counter_container& triplet_counter_container,
                      vecmem::memory_resource* resource);

}  // namespace cuda
}  // namespace traccc
