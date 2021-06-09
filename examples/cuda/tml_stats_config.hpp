/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */
#include "cuda/algorithms/seeding/detail/stats_config.hpp"

#include <algorithm>

namespace traccc {

namespace cuda{
    
class tml_stats_config: public stats_config {
public:
    tml_stats_config(){
	safety_factor = 1.5;
	safety_adder = 10;       	
	par_for_mb_doublets = {1, 28.77, 0.4221};
	par_for_mt_doublets = {1, 19.73, 0.232};
	par_for_triplets = {1, 0, 0.02149};
	par_for_seeds = {0, 0.3431};
    }
    
    size_t get_mid_bot_doublets_size(int n_spM) const {
	return (par_for_mb_doublets(n_spM) + safety_adder) * safety_factor;
    }

    size_t get_mid_top_doublets_size(int n_spM) const {
	return (par_for_mt_doublets(n_spM) + safety_adder) * safety_factor;	
    }

    size_t get_triplets_size(int n_spM) const {
	return (par_for_triplets(n_spM) + safety_adder) * safety_factor;
    }

    size_t get_seeds_size(int n_internal_sp) const{
	return (par_for_seeds(n_internal_sp) + safety_adder) * safety_factor;
    }
    
private:
    scalar safety_factor;
    scalar safety_adder;
    // mid-bot doublets size allocation parameter
    pol2_params par_for_mb_doublets;
    // mid-top doublets size allocation parameter
    pol2_params par_for_mt_doublets;
    // triplets size allocation parameter
    pol2_params par_for_triplets;
    // seeds size allocation parameter
    pol1_params par_for_seeds;    
    
};

} // namespace cuda    
} // namespace traccc
