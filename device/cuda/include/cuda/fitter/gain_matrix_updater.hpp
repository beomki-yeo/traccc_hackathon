/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda/fitter/detail/gain_matrix_updater_helper.hpp>

namespace traccc {
namespace cuda {

template < typename T, int dim, int batch_size >
class gain_matrix_updater{

template<typename track_state_container>
void update(const track_state_container& track_states) {
    m_helper.update();
}
	
private:
    gain_matrix_updater_helper<T, dim, batch_size> m_helper;    
};


}  // namespace cuda
}  // namespace traccc

