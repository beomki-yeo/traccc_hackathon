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
#include <edm/track_parameters.hpp>
#include <limits>

// traccc
#include <edm/detail/transform_bound_to_free.hpp>
#include <edm/detail/transform_free_to_bound.hpp>
#include <propagator/detail/covariance_engine.hpp>
#include <utils/vector_helpers.hpp>

namespace traccc {
    
class eigen_stepper {
    public:
    
    struct __CUDA_ALIGN__(16) state {
        using jacobian_t = typename bound_track_parameters::jacobian_t;
        using covariance_t = typename bound_track_parameters::covariance_t;
        using bound_state_t =
            std::tuple<bound_track_parameters, jacobian_t, double>;
        using curvilinear_state_t =
            std::tuple<curvilinear_track_parameters, jacobian_t, double>;

        state() = default;
	
        state(bound_track_parameters par,
	      host_surface_collection& surfaces,
	      Acts::NavigationDirection ndir = Acts::forward,
	      double ssize = std::numeric_limits<double>::max(),
	      double ssize_cutoff = std::numeric_limits<double>::max(),
	      double stolerance = Acts::s_onSurfaceTolerance)
	
            : q(par.charge()),
              nav_dir(ndir),
              //step_size(ndir * std::abs(ssize)), whyyyy?
              step_size_cutoff(ssize_cutoff),
              tolerance(stolerance) {
	    
            // Get the reference surface for navigation
            const auto& surface = par.reference_surface(surfaces);
            pars = detail::transform_bound_to_free_parameters(par.vector(),
                                                              surface);

            // set the covariance transport flag to true and copy
            covTransport = true;
            cov = par.m_covariance;
	
            ///// ToDo
            // jacToGlobal = surface.boundToFreeJacobian(gctx,par.parameters());	    
        }
	/// Navigation direction, this is needed for searching
        Acts::NavigationDirection nav_dir;
	//double nav_dir;
	
        /// Covariance matrix (and indicator)
        /// associated with the initial error on track parameters
	int covTransport = true;
	//double covTransport = true;
	
        /// The charge as the free vector can be 1/p or q/p
        double q = 1.;

        /// Accummulated path length state
        double path_accumulated = 0.;
	
        /// Adaptive step size of the runge-kutta integration
        double step_size = 10.;

        // cutoff stepsize
        double step_size_cutoff = 0;
	
        /// Last performed step (for overstep limit calculation)
        double previous_step_size = 0.;

        /// The tolerance for the stepping
        double tolerance = Acts::s_onSurfaceTolerance;
	
        size_t max_rk_step_trials = 10000;
	
	/// Internal free vector parameters
        Acts::FreeVector pars = Acts::FreeVector::Zero();
	
        covariance_t cov = covariance_t::Zero();
	
        /// The full jacobian of the transport entire transport
        jacobian_t jacobian = jacobian_t::Identity();
	
        /// Jacobian from local to the global frame
        Acts::BoundToFreeMatrix jac_to_global = Acts::BoundToFreeMatrix::Zero();
	
        /// Pure transport jacobian part from runge kutta integration
        Acts::FreeMatrix jac_transport = Acts::FreeMatrix::Identity();
	
        /// The propagation derivative
        Acts::FreeVector derivative = Acts::FreeVector::Zero();	
	
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

    template <typename surface_t>
    bound_track_parameters bound_state(const surface& surface) {
	// defined at covariance_engine.hpp
        //return detail::bound_state();
    }

    template <typename propagator_state_t>
    static void step(propagator_state_t& state) {

        // state.stepping -> eigen_stepper::state
    }

    template <typename propagator_state_t>
    static __CUDA_HOST_DEVICE__ bool rk4(propagator_state_t& state) {
        return rk4(state.stepping);
    }

    static __CUDA_HOST_DEVICE__ bool rk4(state& state) {

        auto& sd = state.step_data;
	
        Acts::ActsScalar error_estimate = 0.;

        sd.k1 = evaluatek(state, sd.B_first, 0);

        // The following functor starts to perform a Runge-Kutta step of a
        // certain size, going up to the point where it can return an estimate
        // of the local integration error. The results are stated in the local
        // variables above, allowing integration to continue once the error is
        // deemed satisfactory
        const auto try_rk4 = [&](const double& h) -> bool {
				 
            // State the square and half of the step size
            const double h2 = h * h;
            const double half_h = h * 0.5;
            const auto& pos = state.pars.template segment<3>(Acts::eFreePos0);

            const auto& dir = state.pars.template segment<3>(Acts::eFreeDir0);
	    
            // Second Runge-Kutta point
            const Acts::Vector3 pos1 = pos + half_h * dir + h2 * 0.125 * sd.k1;
            // sd.B_middle = getField(state.stepping, pos1);
            sd.k2 = evaluatek(state, sd.B_middle, 1, half_h, sd.k1);

            // Third Runge-Kutta point
            sd.k3 = evaluatek(state, sd.B_middle, 2, half_h, sd.k2);

            // Last Runge-Kutta point
            const Acts::Vector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
            // sd.B_last = getField(state.stepping, pos2);
            sd.k4 = evaluatek(state, sd.B_last, 3, h, sd.k3);

            // Compute and check the local integration error estimate
            // @Todo
            error_estimate = std::max(
                h2 * (sd.k1 - sd.k2 - sd.k3 + sd.k4).template lpNorm<1>(),
                static_cast<Acts::ActsScalar>(1e-20));

            return (error_estimate <= state.tolerance);
        };

        Acts::ActsScalar step_size_scaling = 1.;
        size_t n_step_trials = 0;

        while (!try_rk4(state.step_size)) {
            step_size_scaling = std::min(
                std::max(0.25, std::pow((state.tolerance /
                                         std::abs(2. * error_estimate)),
                                        0.25)),
                4.);

            state.step_size = state.step_size * step_size_scaling;

	    //printf("%f %f \n", state.step_size, step_size_scaling);
	    //	    std::cout << state.step_size << "  " << step_size_scaling << std::endl;

            // Todo: adapted error handling on GPU?
            // If step size becomes too small the particle remains at the
            // initial place
            if (state.step_size * state.step_size <
                state.step_size_cutoff * state.step_size_cutoff) {
                // Not moving due to too low momentum needs an aborter
		printf("step size is too small. will break. \n");
                return false;
            }

            // If the parameter is off track too much or given stepSize is not
            // appropriate
            if (n_step_trials > state.max_rk_step_trials) {
                // Too many trials, have to abort
		printf("too many rk4 trials. will break. \n");
                return false;
            }
            n_step_trials++;
        }

        // Todo: Propagate Time

        auto& h = state.step_size;

        auto pos = state.pars.template segment<3>(Acts::eFreePos0);
        auto dir = state.pars.template segment<3>(Acts::eFreeDir0);

        // Update the track parameters according to the equations of motion
        pos += h * dir + h * h / 6. * (sd.k1 + sd.k2 + sd.k3);

        dir += h / 6. * (sd.k1 + 2. * (sd.k2 + sd.k3) + sd.k4);

        dir /= dir.norm();

        state.derivative.template head<3>() = dir;
        state.derivative.template segment<3>(4) = sd.k4;

        state.path_accumulated += h;
        return true;
    }

    static __CUDA_HOST_DEVICE__ Acts::Vector3 evaluatek(
        const state& state, const Acts::Vector3& bField, const int i = 0,
        const Acts::ActsScalar h = 0.,
        const Acts::Vector3& kprev = Acts::Vector3(0, 0, 0)) {

        Acts::Vector3 knew;
        const auto& qop = state.pars[Acts::eFreeQOverP];
        const auto& dir = state.pars.template segment<3>(Acts::eFreeDir0);

        // First step does not rely on previous data
        if (i == 0) {
            knew = qop * dir.cross(bField);
        } else {
            knew = qop * (dir + h * kprev).cross(bField);
        }
        return knew;
    }

    template <typename propagator_state_t>
    static __CUDA_HOST_DEVICE__ void cov_transport(propagator_state_t& state) {
        cov_transport(state.stepping, state.options.mass);
    }

    static __CUDA_HOST_DEVICE__ void cov_transport(state& state,
                                                   Acts::ActsScalar mass) {
        Acts::FreeMatrix D = Acts::FreeMatrix::Identity();
        const auto& dir = state.pars.template segment<3>(Acts::eFreeDir0);
        const auto& qop = state.pars[Acts::eFreeQOverP];
        Acts::ActsScalar p = abs(1 / qop);

        auto& h = state.step_size;
        const Acts::ActsScalar half_h = h * 0.5;

        auto& sd = state.step_data;

        // For the case without energy loss
        Acts::Vector3 dk1dL = dir.cross(sd.B_first);
        Acts::Vector3 dk2dL = (dir + half_h * sd.k1).cross(sd.B_middle) +
                              qop * half_h * dk1dL.cross(sd.B_middle);
        Acts::Vector3 dk3dL = (dir + half_h * sd.k2).cross(sd.B_middle) +
                              qop * half_h * dk2dL.cross(sd.B_middle);
        Acts::Vector3 dk4dL = (dir + h * sd.k3).cross(sd.B_last) +
                              qop * h * dk3dL.cross(sd.B_last);

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
            dk2dT = qop * vector_helpers::cross(dk2dT, sd.B_middle);
        }

        // std::cout << dk2dT << std::endl;

        Acts::ActsMatrix<3, 3> dk3dT = Acts::ActsMatrix<3, 3>::Identity();
        {
            dk3dT += half_h * dk2dT;
            dk3dT = qop * vector_helpers::cross(dk3dT, sd.B_middle);
        }
        Acts::ActsMatrix<3, 3> dk4dT = Acts::ActsMatrix<3, 3>::Identity();
        {
            dk4dT += h * dk3dT;
            dk4dT = qop * vector_helpers::cross(dk4dT, sd.B_last);
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
        D(3, 7) = h * mass * mass * state.q / (p * std::hypot(1., mass / p));

        state.jac_transport = D * state.jac_transport;
    }
    
    Acts::ActsScalar m_overstep_limit = 0.01;
};

}  // namespace traccc
