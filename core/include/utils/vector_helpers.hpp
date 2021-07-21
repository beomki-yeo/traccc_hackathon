/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/Definitions/Common.hpp"
#include "Acts/EventData/TrackParameters.hpp"

namespace traccc{
namespace vector_helpers{

inline Acts::ActsMatrix<3, 3> cross(const Acts::ActsMatrix<3, 3>& m,
				    const Acts::Vector3& v) {
    Acts::ActsMatrix<3, 3> r;
    r.col(0) = m.col(0).cross(v);
    r.col(1) = m.col(1).cross(v);
    r.col(2) = m.col(2).cross(v);
    
    return r;
}

} // namespace vector_helpers
} // namespace traccc
