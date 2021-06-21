/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithms/seeding/detail/seeding_config.hpp>
#include <algorithms/seeding/detail/statistics.hpp>
#include <algorithms/seeding/doublet_finding.hpp>
#include <algorithms/seeding/seed_filtering.hpp>
#include <algorithms/seeding/triplet_finding.hpp>
#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <iostream>

namespace traccc {

/// Seed finding
struct seed_finding {

    /// Constructor for the seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param isp_container is internal spacepoint container
    seed_finding(seedfinder_config& config,
                 host_internal_spacepoint_container& isp_container)
        : m_doublet_finding(config, isp_container),
          m_triplet_finding(config, isp_container),
          m_isp_container(isp_container) {}

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    host_seed_collection operator()() {
        host_seed_collection seed_collection;
        this->operator()(seed_collection);

        return seed_collection;
    }

    /// Callable operator for the seed finding
    ///
    /// void interface
    ///
    /// @return seed_collection is the vector of seeds per event    
    void operator()(host_seed_collection& seeds) {
        // iterate over grid bins
        for (size_t i = 0; i < m_isp_container.headers.size(); ++i) {
            auto& bin_information = m_isp_container.headers[i];
            auto& spM_collection = m_isp_container.items[i];

	    // multiplet statistics for GPU vector size estimation
            multiplet_statistics stats({0, 0, 0, 0});
            stats.n_spM = spM_collection.size();

            /// iterate over middle spacepoints
            for (size_t j = 0; j < spM_collection.size(); ++j) {
                sp_location spM_location({i, j});

                // middule-bottom doublet search
                auto mid_bot =
                    m_doublet_finding(bin_information, spM_location, true);

                if (mid_bot.first.empty())
                    continue;

                // middule-top doublet search		
                auto mid_top =
                    m_doublet_finding(bin_information, spM_location, false);

                if (mid_top.first.empty())
                    continue;

                host_triplet_collection triplets_per_spM;

                // triplet search from the combinations of two doublets which share middle spacepoint
                for (size_t i = 0; i < mid_bot.first.size(); ++i) {
                    auto& doublet_mb = mid_bot.first[i];
                    auto& lb = mid_bot.second[i];

                    host_triplet_collection triplets = m_triplet_finding(
                        doublet_mb, lb, mid_top.first, mid_top.second);

                    triplets_per_spM.insert(std::end(triplets_per_spM),
                                            triplets.begin(), triplets.end());
                }

		// seed filtering
                m_seed_filtering(m_isp_container, triplets_per_spM, seeds);

                stats.n_mid_bot_doublets += mid_bot.first.size();
                stats.n_mid_top_doublets += mid_top.first.size();
                stats.n_triplets += triplets_per_spM.size();
            }

            m_multiplet_stats.push_back(stats);
        }

        m_seed_stats = seed_statistics({0, 0});
        for (size_t i = 0; i < m_isp_container.headers.size(); ++i) {
            m_seed_stats.n_internal_sp += m_isp_container.items[i].size();
        }
        m_seed_stats.n_seeds = seeds.size();
    }

    std::vector<multiplet_statistics> get_multiplet_stats() {
        return m_multiplet_stats;
    }

    seed_statistics get_seed_stats() { return m_seed_stats; }

   private:
    host_internal_spacepoint_container& m_isp_container;
    doublet_finding m_doublet_finding;
    triplet_finding m_triplet_finding;
    seed_filtering m_seed_filtering;
    seed_statistics m_seed_stats;
    std::vector<multiplet_statistics> m_multiplet_stats;
};
}  // namespace traccc
