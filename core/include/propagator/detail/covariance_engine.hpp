/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/EventData/TrackParameters.hpp"

// traccc
#include <edm/track_parameters.hpp>
#include <edm/detail/transform_free_to_bound.hpp>

namespace traccc {
namespace detail {
    
static __CUDA_HOST_DEVICE__    
void reinitialize_jacobians(Acts::FreeMatrix& free_transport_jacobian,
			    Acts::FreeVector& free_to_path_derivatives,
			    Acts::BoundToFreeMatrix& bound_to_free_jacobian,
			    const Acts::FreeVector& free_vector,
			    const surface& surface) {
    // Reset the jacobians
    free_transport_jacobian = Acts::FreeMatrix::Identity();
    free_to_path_derivatives = Acts::FreeVector::Zero();
    
    // Transform from free to bound parameters
    Acts::BoundVector bound_vector = detail::transform_free_to_bound_parameters(free_vector, surface);

    // Reset the jacobian from local to global    
    bound_to_free_jacobian = surface.bound_to_free_jacobian(bound_vector);
    
}
    
static __CUDA_HOST_DEVICE__    
void bound_to_bound_jacobian(const Acts::FreeVector& free_vector,
			     const Acts::BoundToFreeMatrix& bound_to_free_jacobian,
			     const Acts::FreeMatrix& free_transport_jacobian,
			     const Acts::FreeVector& free_to_path_derivatives,
			     Acts::BoundMatrix& full_transport_jacobian,
			     const surface& surface) {
    
    // Calculate the derivative of path length at the final surface or the
    // point-of-closest approach w.r.t. free parameters
    const Acts::FreeToPathMatrix free_to_path =
	surface.free_to_path_derivative(free_vector);
    // Calculate the jacobian from free to bound at the final surface
    Acts::FreeToBoundMatrix free_to_bound_jacobian =
	surface.free_to_bound_jacobian(free_vector);
    // Calculate the full jacobian from the local/bound parameters at the start
    // surface to local/bound parameters at the final surface
    // @note jac(locA->locB) = jac(gloB->locB)*(1+
    // pathCorrectionFactor(gloB))*jacTransport(gloA->gloB) *jac(locA->gloA)
    
    full_transport_jacobian =
	free_to_bound_jacobian *
	(Acts::FreeMatrix::Identity() + free_to_path_derivatives * free_to_path) *
	free_transport_jacobian * bound_to_free_jacobian;    
}

static __CUDA_HOST_DEVICE__    
void transport_covariance_to_bound(
     Acts::BoundSymMatrix& boundCovariance,
     Acts::BoundMatrix& fullTransportJacobian, Acts::FreeMatrix& freeTransportJacobian,
     Acts::FreeVector& freeToPathDerivatives, Acts::BoundToFreeMatrix& boundToFreeJacobian,
     const Acts::FreeVector& freeParameters, const surface& surface) {
    
    // Calculate the full jacobian from local parameters at the start surface to
    // current bound parameters
    bound_to_bound_jacobian(freeParameters, boundToFreeJacobian,
			    freeTransportJacobian, freeToPathDerivatives,
			    fullTransportJacobian, surface);

    // Apply the actual covariance transport to get covariance of the current
    // bound parameters
    boundCovariance = fullTransportJacobian * boundCovariance *
	fullTransportJacobian.transpose();

    // Reinitialize jacobian components:
    // ->The transportJacobian is reinitialized to Identity
    // ->The derivatives is reinitialized to Zero
    // ->The boundToFreeJacobian is initialized to that at the current surface
    reinitialize_jacobians(freeTransportJacobian,
			   freeToPathDerivatives, boundToFreeJacobian,
			   freeParameters, surface);    
}

static __CUDA_HOST_DEVICE__    
std::tuple<bound_track_parameters, typename bound_track_parameters::jacobian_t, double>
bound_state(Acts::BoundSymMatrix& covarianceMatrix,
	    Acts::BoundMatrix& jacobian,
	    Acts::FreeMatrix& transportJacobian,
	    Acts::FreeVector& derivatives,
	    Acts::BoundToFreeMatrix& boundToFreeJacobian,
	    const Acts::FreeVector& parameters, bool covTransport,
	    double accumulatedPath, const int& surface_id, host_collection<surface>& surfaces) {
    // surface
    const surface& surface = surfaces.items[surface_id];
    
    // Covariance transport
    Acts::BoundSymMatrix cov;
    if (covTransport) {
	// Initialize the jacobian from start local to final local
	jacobian = Acts::BoundMatrix::Identity();
	// Calculate the jacobian and transport the covarianceMatrix to final local.
	// Then reinitialize the transportJacobian, derivatives and the
	// boundToFreeJacobian
	transport_covariance_to_bound(covarianceMatrix, jacobian,
				      transportJacobian, derivatives,
				      boundToFreeJacobian, parameters, surface);
    }
    if (covarianceMatrix != Acts::BoundSymMatrix::Zero()) {
	cov = covarianceMatrix;
    }
    
    // Create the bound parameters
    Acts::BoundVector bv =
	detail::transform_free_to_bound_parameters(parameters, surface);
    
    // Create the bound state (std::move for cov?)
    return std::make_tuple(bound_track_parameters({bv, cov, surface_id}), jacobian, accumulatedPath);
    
}
    

} // namespace detail	
}  // namespace traccc
