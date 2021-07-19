/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <definitions/algebra.hpp>

namespace traccc {

class surface {

    surface(const transform3& transform) { m_transform = transform; }

    transform3 transform() { return m_transform; }

    private:
    transform3 m_transform;
};

}  // namespace traccc
