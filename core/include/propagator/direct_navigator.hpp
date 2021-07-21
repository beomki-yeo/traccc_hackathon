/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "geometry/surface.hpp"

namespace traccc {

template < typename surface_type = surface >    
class direct_navigator {
public:

    using surface_t = surface_type;
    
    struct state {};
};

}  // namespace traccc
