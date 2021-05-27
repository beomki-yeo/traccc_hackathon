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

triplet_finding(seedfinder_config& config, const host_internal_spacepoint_container& isp_container):
    m_config(config),
    m_isp_container(isp_container)
    {
	// calculation of scattering using the highland formula
	// convert pT to p once theta angle is known
	m_config.highland = 13.6 * std::sqrt(m_config.radLengthPerSeed) *
	    (1 + 0.038 * std::log(m_config.radLengthPerSeed));
	float maxScatteringAngle = m_config.highland / m_config.minPt;
	m_config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
	// helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
	// millimeter
	// TODO: change using ACTS units
	m_config.pTPerHelixRadius = 300. * m_config.bFieldInZ;
	m_config.minHelixDiameter2 =
	    std::pow(m_config.minPt * 2 / m_config.pTPerHelixRadius, 2);
	m_config.pT2perRadius =	    
	    std::pow(m_config.highland / m_config.pTPerHelixRadius, 2);	
    }
    
    
void operator()(const doublet mid_bot,
		const host_doublet_collection doublets_mid_top,
		host_triplet_collection& triplets){

    auto spM_idx = mid_bot.sp1;
    auto spM = m_isp_container.items[spM_idx.bin_idx][spM_idx.sp_idx];
    
    float rM = spM.radius();
    float zM = spM.z();
    float varianceRM = spM.varianceR();
    float varianceZM = spM.varianceZ();

    auto lb = mid_bot.lin;
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
    
    for (auto mid_top: doublets_mid_top){
	auto lt = mid_top.lin;

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

	    // inverse diameter is signed depending if the curvature is
	    // positive/negative in phi
	    
	    triplets.push_back({mid_bot.sp2, // bottom
				mid_bot.sp1, // middle
				mid_top.sp2, // top
				B/std::sqrt(S2),
				Im,
				Zob});	    
	}	    
    }

    for (size_t i=0; i<triplets.size(); ++i){
	auto& current_triplet = triplets[i];
	auto& spT_idx = current_triplet.sp3;
	auto& current_spT = m_isp_container.items[spT_idx.bin_idx][spT_idx.sp_idx];
	// if two compatible seeds with high distance in r are found, compatible
	// seeds span 5 layers
	// -> very good seed
	std::vector<float> compatibleSeedR;
	float invHelixDiameter = current_triplet.curvature;
	float lowerLimitCurv = invHelixDiameter - m_filter_config.deltaInvHelixDiameter;
	float upperLimitCurv = invHelixDiameter + m_filter_config.deltaInvHelixDiameter;

	float currentTop_r = current_spT.radius();
	float impact = current_triplet.impact_parameter;
	current_triplet.weight = -(impact * m_filter_config.impactWeightFactor);   

	for (size_t j=0; j<triplets.size(); ++j){
	    if (i == j) {
		continue;
	    }
	    auto& other_triplet = triplets[i];
	    auto& other_spT_idx = other_triplet.sp3;		    
	    auto& other_spT = m_isp_container.items[other_spT_idx.bin_idx][other_spT_idx.sp_idx];
	    // compared top SP should have at least deltaRMin distance
	    float otherTop_r = other_spT.radius();
	    float deltaR = currentTop_r - otherTop_r;
	    if (std::abs(deltaR) < m_filter_config.deltaRMin) {
		continue;
	    }
	    
	    // curvature difference within limits?
	    // TODO: how much slower than sorting all vectors by curvature
	    // and breaking out of loop? i.e. is vector size large (e.g. in jets?)
	    if (other_triplet.curvature < lowerLimitCurv) {
		continue;
	    }
	    if (other_triplet.curvature > upperLimitCurv) {
		continue;
	    }
	    
	    bool newCompSeed = true;
	    for (float previousDiameter : compatibleSeedR) {
		// original ATLAS code uses higher min distance for 2nd found compatible
		// seed (20mm instead of 5mm)
		// add new compatible seed only if distance larger than rmin to all
		// other compatible seeds
		if (std::abs(previousDiameter - otherTop_r) < m_filter_config.deltaRMin) {
		    newCompSeed = false;
		    break;
		}
	    }
	    
	    if (newCompSeed) {
		compatibleSeedR.push_back(otherTop_r);
		current_triplet.weight += m_filter_config.compatSeedWeight;
	    }
	    
	    if (compatibleSeedR.size() >= m_filter_config.compatSeedLimit) {
		break;
	    }	   
	}
    }
}
    
private:
    seedfinder_config m_config;
    seedfilter_config m_filter_config;
    const host_internal_spacepoint_container& m_isp_container;
};

} // namespace traccc
