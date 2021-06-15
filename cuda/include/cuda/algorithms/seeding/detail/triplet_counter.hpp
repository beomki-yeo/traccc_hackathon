/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithms/seeding/detail/doublet.hpp>

namespace traccc {
namespace cuda {

struct triplet_counter {
    doublet mid_bot_doublet;
    size_t n_triplets = 0;
};

/// Container of triplet_counter belonging to one detector module
template <template <typename> class vector_t>
using triplet_counter_collection = vector_t<triplet_counter>;

/// Convenience declaration for the triplet_counter collection type to use in
/// host code
using host_triplet_counter_collection =
    triplet_counter_collection<vecmem::vector>;

/// Convenience declaration for the triplet_counter collection type to use in
/// device code
using device_triplet_counter_collection =
    triplet_counter_collection<vecmem::device_vector>;

/// Convenience declaration for the triplet_counter container type to use in
/// host code
using host_triplet_counter_container =
    host_container<unsigned int, triplet_counter>;

/// Convenience declaration for the triplet_counter container type to use in
/// device code
using device_triplet_counter_container =
    device_container<unsigned int, triplet_counter>;

/// Convenience declaration for the triplet_counter container data type to use
/// in host code
using triplet_counter_container_data =
    container_data<unsigned int, triplet_counter>;

/// Convenience declaration for the triplet_counter container buffer type to use
/// in host code
using triplet_counter_container_buffer =
    container_buffer<unsigned int, triplet_counter>;

/// Convenience declaration for the triplet_counter container view type to use
/// in host code
using triplet_counter_container_view =
    container_view<unsigned int, triplet_counter>;

}  // namespace cuda
}  // namespace traccc
