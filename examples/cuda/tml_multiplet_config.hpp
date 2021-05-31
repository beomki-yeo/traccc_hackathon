/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */
#include "cuda/algorithms/seeding/detail/multiplet_config.hpp"

#include <algorithm>

namespace traccc {

namespace cuda{
    
class tml_multiplet_config: public multiplet_config {
public:
    tml_multiplet_config(){
	safety_factor = 2;
	safety_adder = 10;
	par_for_mb_doublets = {1, 25.85, 0.4009};
	par_for_mt_doublets = {1, 18.36, 0.2184};
	par_for_triplets = {1, 0, 0.01939};	
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
    
private:
    scalar safety_factor;
    scalar safety_adder;
    // mid-bot doublets size allocation parameter
    pol2_params par_for_mb_doublets = {1, 25.85, 0.4009};
    // mid-top doublets size allocation parameter
    pol2_params par_for_mt_doublets = {1, 18.36, 0.2184};
    // triplets size allocation parameter
    pol2_params par_for_triplets = {1, 0, 0.01939};    

};

} // namespace cuda    
} // namespace traccc
