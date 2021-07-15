/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <vector>

#include "container.hpp"
#include "cell.hpp"
#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"
#include "utils/arch_qualifiers.hpp"

// Acts
#include "Acts/Definitions/TrackParametrization.hpp"

namespace traccc {

/// A measurement definition:
/// fix to two-dimensional here

// parameter dimension definition
    
struct measurement {
    
    point2 local = {0., 0.};
    variance2 variance = {0., 0.};

    using projector_t = Acts::ActsMatrix<2,Acts::eBoundSize>;

    projector_t projector(){
	return projector_t::Identity();
    }

    __CUDA_HOST_DEVICE__
    auto& get_local() { return local; }

    __CUDA_HOST_DEVICE__
    auto& get_variance() { return variance; }    
        
};

/// Container of measurements belonging to one detector module
template <template <typename> class vector_t>
using measurement_collection = vector_t<measurement>;

/// Convenience declaration for the measurement collection type to use in host
/// code
using host_measurement_collection = measurement_collection<vecmem::vector>;
/// Convenience declaration for the measurement collection type to use in device
/// code
using device_measurement_collection =
    measurement_collection<vecmem::device_vector>;

/// Convenience declaration for the measurement container type to use in host
/// code
using host_measurement_container = host_container<cell_module, measurement>;

/// Convenience declaration for the measurement container type to use in device
/// code
using device_measurement_container = device_container<cell_module, measurement>;
/// Convenience declaration for the measurement container data type to use in
/// host code
using measurement_container_data = container_data<cell_module, measurement>;

/// Convenience declaration for the measurement container buffer type to use in
/// host code
using measurement_container_buffer = container_buffer<cell_module, measurement>;

/// Convenience declaration for the measurement container view type to use in
/// host code
using measurement_container_view = container_view<cell_module, measurement>;

}  // namespace traccc
