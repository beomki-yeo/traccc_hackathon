/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts
#include "Acts/Definitions/Common.hpp"
#include "Acts/EventData/TrackParameters.hpp"

// vecmem
#include <vecmem/memory/host_memory_resource.hpp>

// traccc
#include <propagator/direct_navigator.hpp>
#include <propagator/eigen_stepper.hpp>
#include <propagator/propagator.hpp>
#include <propagator/propagator_options.hpp>

#include "edm/collection.hpp"

namespace traccc {
namespace cuda {

template <typename stepper_t, typename navigator_t>
class propagator final {

    public:
    template <typename propagator_options_t>
    struct state {

        state(const int n_tracks, vecmem::memory_resource* mr)
            : m_mr(mr),
              options(
                  {typename host_collection<propagator_options_t>::item_vector(
                      n_tracks, mr)}),
              stepping({typename host_collection<
                  typename stepper_t::state>::item_vector(n_tracks, mr)}),
              navigation({typename host_collection<
                  typename navigator_t::state>::item_vector(n_tracks, mr)})

        {}

        /// These are the options - provided for each propagation step
        host_collection<propagator_options_t> options;

        /// Stepper state container - internal states of the Stepper
        host_collection<typename stepper_t::state> stepping;

        /// Navigator state container - internal states of the Navigator
        host_collection<typename navigator_t::state> navigation;

        vecmem::memory_resource* m_mr;
    };

    template <typename propagator_state_t, typename surface_t>
    void propagate(propagator_state_t& state,
                   host_collection<surface_t>& surfaces,
                   vecmem::memory_resource* resource) {

        return propagate(state.options, state.stepping, state.navigation,
                         surfaces, resource);
    }

    template <typename propagator_options_t, typename surface_t>
    void propagate(host_collection<propagator_options_t>& options,
                   host_collection<typename stepper_t::state>& stepping,
                   host_collection<typename navigator_t::state>& navigation,
                   host_collection<surface_t>& surfaces,
                   vecmem::memory_resource* resource);

    private:
    stepper_t m_stepper;
    navigator_t m_navigator;
};

}  // namespace cuda
}  // namespace traccc
