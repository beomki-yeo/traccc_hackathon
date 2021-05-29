/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */


#pragma once

#include <edm/internal_spacepoint.hpp>
#include <algorithms/seeding/detail/doublet.hpp>
#include <algorithms/seeding/detail/triplet.hpp>

namespace traccc{

// helper function used for both cpu and gpu
struct triplet_finding_helper{
    
static bool isCompatible(const internal_spacepoint<spacepoint>& spM,
			 const lin_circle& lb,
			 const lin_circle& lt,
			 const seedfinder_config& config,
			 const scalar& iSinTheta2,
			 const scalar& scatteringInRegion2,			 
			 scalar& curvature,
			 scalar& impact_parameter);    
};
    
bool triplet_finding_helper::isCompatible(const internal_spacepoint<spacepoint>& spM,
					  const lin_circle& lb,
					  const lin_circle& lt,
					  const seedfinder_config& config,
					  const scalar& iSinTheta2,
					  const scalar& scatteringInRegion2,
					  scalar& curvature,
					  scalar& impact_parameter){   
    
    // add errors of spB-spM and spM-spT pairs and add the correlation term
    // for errors on spM
    float error2 = lt.Er + lb.Er +
	2 * (lb.cotTheta * lt.cotTheta * spM.varianceR() + spM.varianceZ()) *
	lb.iDeltaR * lt.iDeltaR;
    
    float deltaCotTheta = lb.cotTheta - lt.cotTheta;
    float deltaCotTheta2 = deltaCotTheta * deltaCotTheta;
    float error;
    float dCotThetaMinusError2;

    // if the error is larger than the difference in theta, no need to
    // compare with scattering	
    if (deltaCotTheta2 - error2 > 0) {
	deltaCotTheta = std::abs(deltaCotTheta);
	// if deltaTheta larger than the scattering for the lower pT cut, skip
	error = std::sqrt(error2);
	dCotThetaMinusError2 =
	    deltaCotTheta2 + error2 - 2 * deltaCotTheta * error;
	// avoid taking root of scatteringInRegion
	// if left side of ">" is positive, both sides of unequality can be
	// squared
	// (scattering is always positive)
	if (dCotThetaMinusError2 > scatteringInRegion2) {
	    return false;
	}
    }
    
    // protects against division by 0
    float dU = lt.U - lb.U;	
    if (dU == 0.) {
	return false;
    }

    // A and B are evaluated as a function of the circumference parameters
    // x_0 and y_0
    float A = (lt.V - lb.V) / dU;
    float S2 = 1. + A * A;
    float B = lb.V - A * lb.U;
    float B2 = B * B;
    // sqrt(S2)/B = 2 * helixradius
    // calculated radius must not be smaller than minimum radius
    if (S2 < B2 * config.minHelixDiameter2) {
	return false;
    }

    
    // 1/helixradius: (B/sqrt(S2))*2 (we leave everything squared)
    float iHelixDiameter2 = B2 / S2;
    // calculate scattering for p(T) calculated from seed curvature
    float pT2scatter = 4 * iHelixDiameter2 * config.pT2perRadius;
    // if pT > maxPtScattering, calculate allowed scattering angle using
    // maxPtScattering instead of pt.
    float pT = config.pTPerHelixRadius * std::sqrt(S2 / B2) / 2.;
    if (pT > config.maxPtScattering) {
	float pTscatter = config.highland / config.maxPtScattering;
	pT2scatter = pTscatter * pTscatter;
    }
    // convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta)
    // from rad to deltaCotTheta
    float p2scatter = pT2scatter * iSinTheta2;
	// if deltaTheta larger than allowed scattering for calculated pT, skip
    if ((deltaCotTheta2 - error2 > 0) &&
	(dCotThetaMinusError2 >
	 p2scatter * config.sigmaScattering * config.sigmaScattering)) {
	return false;
    }

    // calculate curvature
    curvature = B/std::sqrt(S2);
    
    // A and B allow calculation of impact params in U/V plane with linear
    // function
    // (in contrast to having to solve a quadratic function in x/y plane)
    impact_parameter = std::abs((A - B * spM.radius()) * spM.radius());

    if (impact_parameter > config.impactMax){
	return false;
    }

    return true;
}

    
} // namespace traccc
