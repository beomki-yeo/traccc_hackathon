/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

struct void_propagator_options {

    /// The mass for the particle for (eventual) material integration
    double mass = 105.6583755 * Acts::UnitConstants::MeV;
    
    /// Default constructor
    void_propagator_options() = default;

};

}
