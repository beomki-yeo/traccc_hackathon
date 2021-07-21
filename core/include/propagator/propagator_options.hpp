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
#include "propagator/detail/void_aborter.hpp"
#include "propagator/detail/void_actor.hpp"

namespace traccc {

/// @brief Options for propagate() call
///
template <typename action_t, typename aborter_t>
struct propagator_options {
    using action_type = action_t;

    /// Default constructor
    propagator_options() = default;

    /// PropagatorOptions copy constructor
    propagator_options(const propagator_options<action_t, aborter_t> &po) =
        default;

    /// Propagation direction
    // NavigationDirection direction = forward;

    /// The |pdg| code for (eventual) material integration - muon default
    int absPdgCode = 13;

    /// The mass for the particle for (eventual) material integration
    double mass = 105.6583755 * Acts::UnitConstants::MeV;

    /// Maximum number of steps for one propagate call
    unsigned int maxSteps = 10000;

    /// Maximum number of Runge-Kutta steps for the stepper step call
    unsigned int maxRungeKuttaStepTrials = 10000;

    /// Absolute maximum step size
    double maxStepSize = std::numeric_limits<double>::max();

    /// Absolute maximum path length
    double pathLimit = std::numeric_limits<double>::max();

    /// Required tolerance to reach target (surface, pathlength)
    double targetTolerance = Acts::s_onSurfaceTolerance;

    // Configurations for Stepper
    /// Tolerance for the error of the integration
    double tolerance = 1e-4;

    /// Cut-off value for the step size
    double stepSizeCutOff = 0.;

    /// The single actor
    action_t action;

    /// The single aborter
    aborter_t aborter;

    /// The navigator initializer
    // DirectNavigatorInitializer initializer;
};

}  // namespace traccc
