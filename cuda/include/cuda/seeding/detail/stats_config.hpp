/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

class stats_config {
   public:
    virtual ~stats_config() = default;

    virtual size_t get_mid_bot_doublets_size(int n_spM) const = 0;

    virtual size_t get_mid_top_doublets_size(int n_spM) const = 0;

    virtual size_t get_triplets_size(int n_spM) const = 0;

    virtual size_t get_seeds_size(int n_internal_sp) const = 0;
};

}  // namespace traccc
