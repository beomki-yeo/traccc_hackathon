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

// std
#include <limits>

#include "propagator/eigen_stepper_impl.hpp"

namespace traccc {

class eigen_stepper {
    public:
    struct state {
        using jacobian_t = typename bound_track_parameters::jacobian_t;
        using covariance_t = typename bound_track_parameters::covariance_t;
        using bound_state_t =
            std::tuple<bound_track_parameters, jacobian_t, double>;
        using curvilinear_state_t =
            std::tuple<curvilinear_track_parameters, jacobian_t, double>;

        state() = default;

        explicit state(const bound_track_parameters& par,
                       host_surface_collection& surfaces,
                       Acts::NavigationDirection ndir = Acts::forward,
                       double ssize = std::numeric_limits<double>::max(),
                       double stolerance = Acts::s_onSurfaceTolerance)
            : q(par.charge()),
              nav_dir(ndir),
              step_size(ndir * std::abs(ssize)),
              tolerance(stolerance) {

            pars.template segment<3>(Acts::eFreePos0) = par.position(surfaces);
            pars.template segment<3>(Acts::eFreeDir0) = par.unit_direction();
            pars[Acts::eFreeTime] = par.time();
            pars[Acts::eFreeQOverP] = par.qop();

            // Get the reference surface for navigation
            const auto& surface = par.reference_surface(surfaces);
            // set the covariance transport flag to true and copy
            covTransport = true;
            cov = par.m_covariance;

            ///// ToDo
            // jacToGlobal = surface.boundToFreeJacobian(gctx,par.parameters());
        }

        /// Internal free vector parameters
        Acts::FreeVector pars = Acts::FreeVector::Zero();

        /// The charge as the free vector can be 1/p or q/p
        double q = 1.;

        /// Covariance matrix (and indicator)
        /// associated with the initial error on track parameters
        bool covTransport = true;
        covariance_t cov = covariance_t::Zero();

        /// Navigation direction, this is needed for searching
        Acts::NavigationDirection nav_dir;

        /// The full jacobian of the transport entire transport
        jacobian_t jacobian = jacobian_t::Identity();

        /// Jacobian from local to the global frame
        Acts::BoundToFreeMatrix jac_to_global = Acts::BoundToFreeMatrix::Zero();

        /// Pure transport jacobian part from runge kutta integration
        Acts::FreeMatrix jac_transport = Acts::FreeMatrix::Identity();

        /// The propagation derivative
        Acts::FreeVector derivative = Acts::FreeVector::Zero();

        /// Accummulated path length state
        double path_accumulated = 0.;

        /// Adaptive step size of the runge-kutta integration
        double step_size = 0.;

        /// Last performed step (for overstep limit calculation)
        double previous_step_size = 0.;

        /// The tolerance for the stepping
        double tolerance = Acts::s_onSurfaceTolerance;

        /// @brief Storage of magnetic field and the sub steps during a RKN4
        /// step
        struct {
            /// Magnetic field evaulations
            Acts::Vector3 B_first, B_middle, B_last;
            /// k_i of the RKN4 algorithm
            Acts::Vector3 k1, k2, k3, k4;
            /// k_i elements of the momenta
            std::array<double, 4> kQoP;
        } step_data;
    };

    state make_state(const bound_track_parameters& par,
                     host_surface_collection& surfaces,
                     Acts::NavigationDirection ndir = Acts::forward,
                     double ssize = std::numeric_limits<double>::max(),
                     double stolerance = Acts::s_onSurfaceTolerance) const {
        return state(par, surfaces, ndir, ssize, stolerance);
    }

    template <typename propagator_state_t>
    void step(propagator_state_t& state) {

	// state.stepping -> eigen_stepper::state

    }

    template <typename propagator_state_t>
    void cov_transport(propagator_state_t& state) {

	eigen_stepper_impl::cov_transport(state.state);

    }
    
};

}  // namespace traccc
