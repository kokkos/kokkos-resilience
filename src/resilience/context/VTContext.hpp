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
#ifndef INC_KOKKOS_RESILIENCE_VTCONTEXT_HPP
#define INC_KOKKOS_RESILIENCE_VTCONTEXT_HPP

#include <stdexcept>
#include <optional>
#include <type_traits>
#include <vt/vt.h>
#include <checkpoint/checkpoint.h>
#include "ContextBase.hpp"
#include "resilience/util/VTUtil.hpp"

namespace KokkosResilience {
  class VTContext;
  using VTContextProxy = Util::VT::VTObj<VTContext>;
  using VTContextElmProxy = Util::VT::VTObjElm<VTContext>;
}

namespace KokkosResilience::Detail {
  enum VTAction {
    REPORT_VERSION,
    REGISTER_PARENT_DEPENDENCY,
    MARK_MODIFIED,
    RESTART
  };

  struct VTActionMsg {
    VTActionMsg(VTContextElmProxy sender, VTAction action, int arg)
      : sender(sender), action(action), arg(arg) {};

    VTContextElmProxy sender;
    VTAction action;
    int arg;
  };

  struct VTProxyHolder : public KokkosResilience::Util::VT::ProxyID {
    using ProxyID = KokkosResilience::Util::VT::ProxyID;

    VTProxyHolder() {
      assert(false && "VTProxyHolder default constructor exists for Magistrate serialization, but should not be used!");
    }

    template<typename ProxyT>
    VTProxyHolder(ProxyT);

    template<typename SerializerT>
    void serialize(SerializerT& s);
    //Trait for writing dependencies map when serializing
    struct CheckpointDeps {};

    //Type-erasure functions lambdas
    std::function<void(VTActionMsg)> send = nullptr;
    std::function<bool()>            is_local = nullptr;
    //Message another index of my group to restart them
    std::function<void(VTContext*, uint64_t)> register_other;

    //Who and at what required version.
    std::unordered_map<ProxyID, int> dependencies;

    //Made unique per element w/ index info
    std::string label;

    int checkpointed_version = -1;
    int recovered_version = -1;

    //Useful for asynchronous checkpoint/recovery.
    vt::EpochType epoch;

    //Only elements get a registration
    std::optional<Registration> registration;
  };
}

namespace KokkosResilience {
  class VTContext : public ContextBase {
  public:
    explicit VTContext(const std::string& config_file)
        : ContextBase(config_file, vt::theContext()->getNode()), 
          contexts_proxy(vt::theObjGroup()->makeCollective(this, "kr::VTContext")) {
      next_epoch = vt::theTerm()->makeEpochCollective("kr::VTContext initial checkpoint epoch");
    }
   
    using ProxyID       = KokkosResilience::Util::VT::ProxyID;
    using VTProxyHolder = Detail::VTProxyHolder;
    using VTAction      = Detail::VTAction;
    using VTActionMsg   = Detail::VTActionMsg;
   
    VTContext(const VTContext &)     = delete;
    VTContext(VTContext &&) noexcept = default;
   
    VTContext &operator=(const VTContext &) = delete;
    VTContext &operator=(VTContext &&) noexcept = default;
   
    virtual ~VTContext() {
      //Wait for next_epoch to finish being created, then finish it.
      vt::theSched()->runSchedulerWhile([&]{return next_epoch == vt::no_epoch;});
      vt::theTerm()->finishedEpoch(next_epoch);

      vt::theObjGroup()->destroyCollective(contexts_proxy);
    }
   
    bool restart_available(const std::string &label, int version) override {
      return m_backend->restart_available(label, version);
    }
   
    void restart(const std::string &label, int version,
                 const std::unordered_set<Registration> &members) override {
      begin_operation();

      m_backend->restart(label, version, members);
      restart_proxies();

      end_operation();
    }
   
    void checkpoint(const std::string &label, int version,
                    const std::unordered_set<Registration> &members) override {
      begin_operation();

      checkpoint_proxies();
      m_backend->checkpoint(label, version, members);

      end_operation();
    }
   
    int latest_version(const std::string &label) const noexcept override {
      return m_backend->latest_version(label);
    }
    
    void reset() override { m_backend->reset(); }

    //Handles initialization for new holders
    template<typename T>
    VTProxyHolder& get_holder(T proxy, bool mark_modified = true);

  private:
    //Start a new resilience operation
    //  Manages the epochs, waiting for previous, beginning new, etc.
    void begin_operation();
    void end_operation();

    //Checkpoint/recover actual data and dependencies of proxies
    void checkpoint_proxies();

    //Recursive restart which manages potentially non-registered proxies
    //  assume_collective to assume non-local proxies are already known by their 
    //  local context to need to recover
    void restart_proxy(ProxyID proxy, int version, bool is_remote_request = false);
    void restart_proxies();

    //Handle marking element or group as modified and the possible
    //remote operations required.
    template<typename ProxyT>
    void mark_modified(ProxyT& proxy, VTProxyHolder& holder, 
        bool is_remote_request = false);

    
    //Send an action to be executed before the next checkpoint/recovery function
    //can begin.
    void msg_before_checkpoint(VTProxyHolder& holder, VTAction action, int arg = 0);
    template<auto func, typename... Args>
    void msg_before_checkpoint(VTContextElmProxy dest, Args... args);
    template<auto func, typename... Args>
    void msg_before_checkpoint(VTContextProxy dest, Args... args);
   
    template<typename ProxyT, typename GroupProxyT>
    void action_handler(ProxyT elm, GroupProxyT group, const VTActionMsg& msg);

  public:
    //Manage globally tracking group proxies.
    template<typename GroupProxyT>
    void register_group(GroupProxyT group, bool should_mark_modified);
    void restart_group(ProxyID group, int version);
    
    //For recovering unregistered entities, manually generate an element
    //registration from a reference to the group (potentially gathered from
    //some other registered element).
    template<typename GroupProxyT>
    void register_element(GroupProxyT group, uint64_t index_bits);
    
    //Remote context notifiying us
    template<typename ProxyT>
    void remotely_modified(ProxyT proxy);

    //Remote context responding to query
    template<typename ProxyT>
    void set_checkpointed_version(ProxyT proxy, int version);

    //Gathers proxies, then calls action_handler
    template<typename ObjT>
    static void col_action_handler(ObjT* obj, VTActionMsg msg);
    template<typename ObjT>
    static void obj_action_handler(ObjT* obj, VTActionMsg msg);

  private:
    using ProxyMap = std::unordered_map<ProxyID, VTProxyHolder>;
    ProxyMap proxy_registry;

    //Map from a collection/objgroup to some registered proxy within it,
    //for reconstructing as needed.
    std::unordered_map<ProxyID, ProxyID> groups;
   
    //Local proxies known to have been changed since last checkpoint
    std::unordered_set<ProxyID> modified_proxies;

    vt::EpochType cur_epoch = vt::no_epoch;
    vt::EpochType next_epoch;

    //Index of which resilience epoch we're ready to begin handling
    //requests for (e.g. have we updated checkpointed_version for our
    //locally modified proxies for a given checkpoint attempt).
    int prepared_epoch = 0;

    VTContextProxy contexts_proxy;
    VTContextElmProxy m_proxy = contexts_proxy[m_pid];
  };
}

#include "VTContext.impl.hpp"
#endif // INC_KOKKOS_RESILIENCE_VTCONTEXT_HPP
