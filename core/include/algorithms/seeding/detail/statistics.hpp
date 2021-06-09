/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc{
    
struct multiplet_statistics{
    size_t n_spM;
    size_t n_mid_bot_doublets;
    size_t n_mid_top_doublets;
    size_t n_triplets;
};

struct seed_statistics{
    size_t n_internal_sp;
    size_t n_seeds;
};
    
}    
