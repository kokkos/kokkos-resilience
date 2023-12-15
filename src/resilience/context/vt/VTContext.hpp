/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */

#ifndef INC_KOKKOS_RESILIENCE_CONTEXT_VT_VTCONTEXT_HPP
#define INC_KOKKOS_RESILIENCE_CONTEXT_VT_VTCONTEXT_HPP

#include <stdexcept>
#include <optional>
#include <any>
#include <type_traits>
#include <unordered_map>
#include <deque>

#include <vt/vt.h>


#include "resilience/context/ContextBase.hpp"
#include "resilience/Config.hpp"
#include "common.hpp"
#include "ProxyHolder.hpp"
#include "ProxyMap.hpp"

namespace KokkosResilience::Context::VT {
  class VTContext : public ContextBase {
  public:
    explicit VTContext(const std::string& config_file);
   
    VTContext(const VTContext &)     = delete;
    VTContext(VTContext &&) noexcept = default;
   
    VTContext &operator=(const VTContext &) = delete;
    VTContext &operator=(VTContext &&) noexcept = default;
   
    virtual ~VTContext();
   
    bool restart_available(const std::string &label, int version) override {
      //TODO: Also recover this version's required proxy versions and 
      //     verify they're available?
      return m_backend->restart_available(label, version);
    }

   
    void restart(const std::string &label, int version,
                 const std::unordered_set<Registration> &members) override;
   
    void checkpoint(const std::string &label, int version,
                    const std::unordered_set<Registration> &members) override;
   
    int latest_version(const std::string &label) const noexcept override {
      return m_backend->latest_version(label);
    }
    
    void reset_impl() override { /* TODO */ }
    
    void register_member(Registration member, Region region) override;

    void deregister_member(Registration member, Region region) override;

    void enter_region(Region region, int version) override;

    void exit_region(Region region, int version) override;

    //Keep a record of recent VTProxy Registrations to proxy ID,
    //for managing (de)registering members that are actually proxies.
    template<typename T>
    void add_reg_mapping(size_t hash, T proxy);

    template<typename T>
    ProxyHolder& get_holder(T proxy);
    
  private:
    //Register as a ContextBase member
    template<typename ProxyT>
    void register_proxy(ProxyT proxy, std::string& region_label);
    template<typename ProxyT>
    void deregister_proxy(ProxyT proxy, std::string& region_label);
    
    //Checkpoint/recover actual data and dependencies of proxies
    //populates checkpoint_epoch
    void checkpoint_proxies(const std::string& label, const int version);
    void checkpoint_proxy(ProxyHolder& holder, vt::EpochType epoch);

    //Recursively traverse proxy and its dependencies
    template<typename ProxyT>
    void restart_proxy(ProxyT proxy, ProxyHolder& holder);
    void restart_proxies(const std::string& label, const int version);

    size_t get_config_max_iter_offset();

  protected:
    friend ProxyHolder;
    friend ProxyMap;
    
    template<typename ProxyT>
    void init_holder(ProxyT proxy, ProxyHolder& holder);
    
    template<typename ProxyT>
    std::any action_handler(
        ProxyT proxy,
        ProxyHolder& holder,
        ProxyAction action,
        std::any arg,
        bool remote_request = false
    );
    
    
  private:
    //If ProxyT is an element, send to element to find correct local context
    //Else, broadcast to contexts.
    template<typename ProxyT, typename ArgT = void*>
    void send_action(
        ProxyT proxy,
        ProxyAction action,
        const ArgT& arg = {}
    );

    //Or send to a specific rank's context
    template<typename ProxyT, typename ArgT = void*>
    void send_action(
        int dest,
        ProxyT proxy,
        ProxyAction action,
        const ArgT& arg = {}
    );

  public:

    //Handle action addressed to element, passing back to action_handler
    template<typename ProxyT, typename ObjT, typename ArgT>
    static void remote_action_handler(
        ObjT* unused,
        VTContextProxy ctx_proxy,
        ProxyT proxy, 
        ProxyAction action,
        ArgT arg
    );
    //As above, but addressed to context
    template<typename ProxyT, typename ArgT>
    void remote_action_handler(
        ProxyT proxy,
        ProxyAction action,
        ArgT arg
    );

    static vt::trace::UserEventIDType checkpoint_region;
    static vt::trace::UserEventIDType checkpoint_wait;
    static vt::trace::UserEventIDType serialize_proxy;
    static vt::trace::UserEventIDType offset_region;
    static vt::trace::UserEventIDType offset_wait;

  protected: 
    //Handle element migrations/insertions/deletions
    using ElementEvent = vt::vrt::collection::listener::ElementEventEnum;
    void handle_element_event(ProxyID proxy, ElementEvent event);

  private:
    //Handle elements from during a phase that had to have handling delayed.
    void handle_pending_events();

    ProxyMap holders;

    //Local proxies known to have been changed since last checkpoint
    std::unordered_set<ProxyID> modified_proxies;

    //Delay handling migrations due to phase balancing until phase ends.
    std::deque<std::pair<ProxyID, ElementEvent>> pending_element_events = {};

    vt::phase::PhaseHookID phase_begin_hookid, phase_end_hookid;
    bool in_phase = false;
    
    
    //Epoch for all proxy element checkpoints finished
    vt::EpochType checkpoint_epoch = vt::no_epoch;

    //Epoch for offset iter, checkpoint_epoch not guaranteed created until
    //this is finished
    const size_t max_iteration_offset = 0;
    vt::EpochType offset_iter_epoch = vt::no_epoch;

    VTContextProxy contexts_proxy;
    VTContextElmProxy m_proxy = contexts_proxy[m_pid];
  };
}

#include "VTContext.impl.hpp"
#endif // INC_KOKKOS_RESILIENCE_VTCONTEXT_HPP
