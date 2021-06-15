/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#include <algorithm>

#include "algorithms/seeding/detail/experiment_cuts.hpp"

namespace traccc {
class atlas_cuts : public experiment_cuts {
   public:
    void seedWeight(scalar& triplet_weight,
                    const internal_spacepoint<spacepoint>& spB,
                    const internal_spacepoint<spacepoint>& spM,
                    const internal_spacepoint<spacepoint>& spT) const;

    bool singleSeedCut(const scalar& triplet_weight,
                       const internal_spacepoint<spacepoint>& spB,
                       const internal_spacepoint<spacepoint>& spM,
                       const internal_spacepoint<spacepoint>& spT) const;

    host_seed_collection cutPerMiddleSP(
        const host_seed_collection& seeds) const;
};

void atlas_cuts::seedWeight(scalar& triplet_weight,
                            const internal_spacepoint<spacepoint>& spB,
                            const internal_spacepoint<spacepoint>& spM,
                            const internal_spacepoint<spacepoint>& spT) const {
    float weight = 0;

    if (spB.radius() > 150) {
        weight = 400;
    }
    if (spT.radius() < 150) {
        weight = 200;
    }

    triplet_weight += weight;
    return;
}

bool atlas_cuts::singleSeedCut(
    const scalar& triplet_weight, const internal_spacepoint<spacepoint>& spB,
    const internal_spacepoint<spacepoint>& spM,
    const internal_spacepoint<spacepoint>& spT) const {
    // bottom

    return !(spB.radius() > 150. && triplet_weight < 380.);
}

host_seed_collection atlas_cuts::cutPerMiddleSP(
    const host_seed_collection& seeds) const {
    host_seed_collection newSeeds;
    if (seeds.size() > 1) {
        newSeeds.push_back(seeds[0]);
        size_t itLength = std::min(seeds.size(), size_t(5));
        // don't cut first element
        for (size_t i = 1; i < itLength; i++) {
            if (seeds[i].weight > 200. || seeds[i].spB.radius() > 43.) {
                newSeeds.push_back(std::move(seeds[i]));
            }
        }
        return newSeeds;
    }
    return seeds;
}

}  // namespace traccc
