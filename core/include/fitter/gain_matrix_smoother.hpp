/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/track_parameters.hpp>
#include <fitter/gain_matrix_smoother_impl.hpp>

// Acts
#include <Acts/Definitions/TrackParametrization.hpp>

namespace traccc {

template <typename track_state_t>
class gain_matrix_smoother {
    public:
    gain_matrix_smoother() = default;

    ~gain_matrix_smoother() = default;

    void operator()(track_state_t& tr_state) {
        gain_matrix_smoother_impl<track_state_t>::smoother(tr_state);
    }
};

}  // namespace traccc
