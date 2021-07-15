/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// acts
#include <Acts/Definitions/TrackParametrization.hpp>

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <cuda/utils/definitions.hpp>


void matmul(int n_matrix,
	    vecmem::data::vector_view< Acts::BoundSymMatrix > A_view,
	    vecmem::data::vector_view< Acts::BoundSymMatrix > B_view,
	    vecmem::data::vector_view< Acts::BoundSymMatrix > C_view,
	    float& elapsed);


void matmul(int n_matrix,
	    vecmem::data::vector_view< Acts::FreeSymMatrix > A_view,
	    vecmem::data::vector_view< Acts::FreeSymMatrix > B_view,
	    vecmem::data::vector_view< Acts::FreeSymMatrix > C_view,
	    float& elapsed);


