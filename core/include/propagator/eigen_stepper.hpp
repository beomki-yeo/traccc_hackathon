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
#include <edm/track_parameters.hpp>
//#include "propagator/eigen_stepper_impl.hpp"

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
    static void step(propagator_state_t& state) {

	// state.stepping -> eigen_stepper::state

    }

    template <typename propagator_state_t>
    static void cov_transport(propagator_state_t& state) {
	cov_transport(state.stepping, state.options.mass);
    }

    template <typename stepper_state_t>
    static __CUDA_HOST_DEVICE__ void cov_transport(stepper_state_t& state,
						   Acts::ActsScalar mass){
	Acts::FreeMatrix D = Acts::FreeMatrix::Identity();
	const auto& dir = state.pars.template segment<3>(Acts::eFreeDir0);
	const auto& qop = state.pars[Acts::eFreeQOverP];
	Acts::ActsScalar p = abs(1/qop);
	
	auto& h = state.step_size;
	const Acts::ActsScalar half_h = h * 0.5;

	auto& sd = state.step_data;

	// For the case without energy loss
	Acts::Vector3 dk1dL = dir.cross(sd.B_first);
	Acts::Vector3 dk2dL = (dir + half_h * sd.k1).cross(sd.B_middle) +
	    qop * half_h * dk1dL.cross(sd.B_middle);
	Acts::Vector3 dk3dL = (dir + half_h * sd.k2).cross(sd.B_middle) +
	    qop * half_h * dk2dL.cross(sd.B_middle);
	Acts::Vector3 dk4dL =
	    (dir + h * sd.k3).cross(sd.B_last) + qop * h * dk3dL.cross(sd.B_last);

	
	// Calculate the dK/dT
	Acts::ActsMatrix<3, 3> dk1dT = Acts::ActsMatrix<3, 3>::Zero();
	{
	    dk1dT(0, 1) = sd.B_first.z();
	    dk1dT(0, 2) = -sd.B_first.y();
	    dk1dT(1, 0) = -sd.B_first.z();
	    dk1dT(1, 2) = sd.B_first.x();
	    dk1dT(2, 0) = sd.B_first.y();
	    dk1dT(2, 1) = -sd.B_first.x();
	    dk1dT *= qop;
	}
	
	Acts::ActsMatrix<3, 3> dk2dT = Acts::ActsMatrix<3, 3>::Identity();
	{
	    dk2dT += half_h * dk1dT;
	    dk2dT = qop * Acts::VectorHelpers::cross(dk2dT, sd.B_middle);
	}

	//std::cout << dk2dT << std::endl;
	
	Acts::ActsMatrix<3, 3> dk3dT = Acts::ActsMatrix<3, 3>::Identity();
	{
	    dk3dT += half_h * dk2dT;
	    dk3dT = qop * Acts::VectorHelpers::cross(dk3dT, sd.B_middle);
	}
	Acts::ActsMatrix<3, 3> dk4dT = Acts::ActsMatrix<3, 3>::Identity();
	{
	    dk4dT += h * dk3dT;
	    dk4dT = qop * Acts::VectorHelpers::cross(dk4dT, sd.B_last);
	}
	// The dF/dT in D
	{
	    auto dFdT = D.block<3, 3>(0, 4);
	    dFdT.setIdentity();
	    dFdT += h / 6. * (dk1dT + dk2dT + dk3dT);
	    dFdT *= h;
	}
	// The dF/dL in D
	{
	    auto dFdL = D.block<3, 1>(0, 7);
	    dFdL = (h * h) / 6. * (dk1dL + dk2dL + dk3dL);
	}
	// The dG/dT in D
	{
	    // dGdx is already initialised as (3x3) zero
	    auto dGdT = D.block<3, 3>(4, 4);
	    dGdT += h / 6. * (dk1dT + 2. * (dk2dT + dk3dT) + dk4dT);
	}
	// The dG/dL in D
	{
	    auto dGdL = D.block<3, 1>(4, 7);
	    dGdL = h / 6. * (dk1dL + 2. * (dk2dL + dk3dL) + dk4dL);
	}
	
	// The dt/d(q/p)
	D(3, 7) = h * mass * mass * state.q /
            (p * std::hypot(1., mass / p));
	
	state.jac_transport = D * state.jac_transport;
    }        
};

}  // namespace traccc
