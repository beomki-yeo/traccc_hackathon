/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// acts
#include <Acts/Definitions/TrackParametrization.hpp>

// VecMem include(s).
#include <cuda/utils/definitions.hpp>
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

template <typename matrix_t>
void matmul(int n_matrix, vecmem::data::vector_view<matrix_t> A_view,
            vecmem::data::vector_view<matrix_t> B_view,
            vecmem::data::vector_view<matrix_t> C_view, float& elapsed);
