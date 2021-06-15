/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithms/seeding/detail/singlet.hpp>

namespace traccc {
namespace cuda {

struct doublet_counter {
    sp_location spM;
    size_t n_mid_bot = 0;
    size_t n_mid_top = 0;
};

/// Container of doublet_counter belonging to one detector module
template <template <typename> class vector_t>
using doublet_counter_collection = vector_t<doublet_counter>;

/// Convenience declaration for the doublet_counter collection type to use in
/// host code
using host_doublet_counter_collection =
    doublet_counter_collection<vecmem::vector>;

/// Convenience declaration for the doublet_counter collection type to use in
/// device code
using device_doublet_counter_collection =
    doublet_counter_collection<vecmem::device_vector>;

/// Convenience declaration for the doublet_counter container type to use in
/// host code
using host_doublet_counter_container =
    host_container<unsigned int, doublet_counter>;

/// Convenience declaration for the doublet_counter container type to use in
/// device code
using device_doublet_counter_container =
    device_container<unsigned int, doublet_counter>;

/// Convenience declaration for the doublet_counter container data type to use
/// in host code
using doublet_counter_container_data =
    container_data<unsigned int, doublet_counter>;

/// Convenience declaration for the doublet_counter container buffer type to use
/// in host code
using doublet_counter_container_buffer =
    container_buffer<unsigned int, doublet_counter>;

/// Convenience declaration for the doublet_counter container view type to use
/// in host code
using doublet_counter_container_view =
    container_view<unsigned int, doublet_counter>;

}  // namespace cuda
}  // namespace traccc
