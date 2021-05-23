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

struct triplet_finding{

triplet_finding(const seedfinder_config& config, const host_internal_spacepoint_container& isp_container):
    m_config(config),
    m_isp_container(isp_container)
    {}
    
host_seed_collection operator()(const std::vector<lin_circle>& bottom_lin_circles,
				const std::vector<lin_circle>& top_lin_circles,
				const std::vector< sp_location >& compat_bottom_id,
				const std::vector< sp_location >& compat_top_id,
				const internal_spacepoint<spacepoint> spM){
    host_seed_collection seed_collection;
    this->operator()(bottom_lin_circles,
		     top_lin_circles,
		     compat_bottom_id,
		     compat_top_id,
		     spM,
		     seed_collection);
    
    return seed_collection;
}
    
void operator()(const std::vector<lin_circle>& bottom_lin_circles,
		const std::vector<lin_circle>& top_lin_circles,
		const std::vector< sp_location >& compat_bottom_id,
		const std::vector< sp_location >& compat_top_id,
		const internal_spacepoint<spacepoint> spM,
		host_seed_collection& seed_collection){

    // create vectors here to avoid reallocation in each loop
    std::vector<internal_spacepoint<spacepoint>> top_sp_vec;
    std::vector<float> curvatures;
    std::vector<float> impact_parameters;

    float rM = spM.radius();
    float zM = spM.z();
    float varianceRM = spM.varianceR();
    float varianceZM = spM.varianceZ();
    
    for (size_t b = 0; b < bottom_lin_circles.size(); b++) {
	auto lb = bottom_lin_circles[b];
	float Zob = lb.Zo;
	float cotThetaB = lb.cotTheta;
	float Vb = lb.V;
	float Ub = lb.U;
	float ErB = lb.Er;
	float iDeltaRB = lb.iDeltaR;
	
	// 1+(cot^2(theta)) = 1/sin^2(theta)
	float iSinTheta2 = (1. + cotThetaB * cotThetaB);
	// calculate max scattering for min momentum at the seed's theta angle
	// scaling scatteringAngle^2 by sin^2(theta) to convert pT^2 to p^2
	// accurate would be taking 1/atan(thetaBottom)-1/atan(thetaTop) <
	// scattering
	// but to avoid trig functions we approximate cot by scaling by
	// 1/sin^4(theta)
	// resolving with pT to p scaling --> only divide by sin^2(theta)
	// max approximation error for allowed scattering angles of 0.04 rad at
	// eta=infinity: ~8.5%
	float scatteringInRegion2 = m_config.maxScatteringAngle2 * iSinTheta2;
	// multiply the squared sigma onto the squared scattering
	scatteringInRegion2 *=
	    m_config.sigmaScattering * m_config.sigmaScattering;

	// clear all vectors used in each inner for loop
	top_sp_vec.clear();
	curvatures.clear();
	impact_parameters.clear();

	for (size_t t = 0; t < top_lin_circles.size(); t++) {
	    auto lt = top_lin_circles[t];

	    // add errors of spB-spM and spM-spT pairs and add the correlation term
	    // for errors on spM
	    float error2 = lt.Er + ErB +
		2 * (cotThetaB * lt.cotTheta * varianceRM + varianceZM) *
		iDeltaRB * lt.iDeltaR;
	    
	    float deltaCotTheta = cotThetaB - lt.cotTheta;
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
		    continue;
		}
	    }

	    // protects against division by 0
	    float dU = lt.U - Ub;
	    if (dU == 0.) {
		continue;
	    }
	    // A and B are evaluated as a function of the circumference parameters
	    // x_0 and y_0
	    float A = (lt.V - Vb) / dU;
	    float S2 = 1. + A * A;
	    float B = Vb - A * Ub;
	    float B2 = B * B;
	    // sqrt(S2)/B = 2 * helixradius
	    // calculated radius must not be smaller than minimum radius
	    if (S2 < B2 * m_config.minHelixDiameter2) {
		continue;
	    }
	    // 1/helixradius: (B/sqrt(S2))*2 (we leave everything squared)
	    float iHelixDiameter2 = B2 / S2;
	    // calculate scattering for p(T) calculated from seed curvature
	    float pT2scatter = 4 * iHelixDiameter2 * m_config.pT2perRadius;
	    // if pT > maxPtScattering, calculate allowed scattering angle using
	    // maxPtScattering instead of pt.
	    float pT = m_config.pTPerHelixRadius * std::sqrt(S2 / B2) / 2.;
	    if (pT > m_config.maxPtScattering) {
		float pTscatter = m_config.highland / m_config.maxPtScattering;
		pT2scatter = pTscatter * pTscatter;
	    }
	    // convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta)
	    // from rad to deltaCotTheta
	    float p2scatter = pT2scatter * iSinTheta2;
	    // if deltaTheta larger than allowed scattering for calculated pT, skip
	    if ((deltaCotTheta2 - error2 > 0) &&
		(dCotThetaMinusError2 >
		 p2scatter * m_config.sigmaScattering * m_config.sigmaScattering)) {
		continue;
	    }

	    // A and B allow calculation of impact params in U/V plane with linear
	    // function
	    // (in contrast to having to solve a quadratic function in x/y plane)
	    float Im = std::abs((A - B * rM) * rM);
	    
	    if (Im <= m_config.impactMax) {
		auto top_loc = compat_top_id[t];
		auto& sp_top = m_isp_container.items[top_loc.bin_idx][top_loc.sp_idx];
		top_sp_vec.push_back(sp_top);
		// inverse diameter is signed depending if the curvature is
		// positive/negative in phi
		curvatures.push_back(B / std::sqrt(S2));
		impact_parameters.push_back(Im);
	    }	    
	} // iterate over top sp

	if (!top_sp_vec.empty()) {
	    /*
	    std::vector<std::pair<
		float, std::unique_ptr<const InternalSeed<external_spacepoint_t>>>>
		sameTrackSeeds;
	    sameTrackSeeds = std::move(m_config.seedFilter->filterSeeds_2SpFixed(
           *compatBottomSP[b], *spM, topSpVec, curvatures, impactParameters, Zob));
	    seedsPerSpM.insert(seedsPerSpM.end(),
			       std::make_move_iterator(sameTrackSeeds.begin()),
			       std::make_move_iterator(sameTrackSeeds.end()));
	    */
	}
    } // iterate over bottom sp

    //m_config.seedFilter->filterSeeds_1SpFixed(seedsPerSpM, outputVec);    
}
    
private:
    seedfinder_config m_config;
    const host_internal_spacepoint_container& m_isp_container;
};

} // namespace traccc
