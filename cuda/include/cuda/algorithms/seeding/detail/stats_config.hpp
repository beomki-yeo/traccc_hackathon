/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc{

struct pol1_params{
    scalar p0;
    scalar p1;
    
    size_t operator()(size_t n) const {
	return size_t( p0+p1*n );
    }
};
    
struct pol2_params{
    scalar p0;
    scalar p1;
    scalar p2;
    
    size_t operator()(size_t n) const {
	return size_t( p0+p1*n+p2*n*n );
    }
};
    
class stats_config{
    
public:

    virtual ~stats_config() = default;

    virtual size_t get_mid_bot_doublets_size(int n_spM) const = 0;

    virtual size_t get_mid_top_doublets_size(int n_spM) const = 0;

    virtual size_t get_triplets_size(int n_spM) const = 0;

    virtual size_t get_seeds_size(int n_internal_sp) const = 0;
    
};    
    
} // namespace traccc
