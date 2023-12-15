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

#ifndef INC_KOKKOS_RESILIENCE_CONTEXT_VT_VTCONTEXT_IMPL_HPP
#define INC_KOKKOS_RESILIENCE_CONTEXT_VT_VTCONTEXT_IMPL_HPP

#include <memory>
#include <string>

#include "common.hpp"

#include "VTContext.hpp"

#include "ProxyHolder.impl.hpp"
#include "ProxyMap.impl.hpp"

namespace KokkosResilience::Context::VT { 
  template<typename ProxyT>
  void VTContext::init_holder(ProxyT proxy, ProxyHolder& holder){
    if constexpr(not is_elm<ProxyT>::value) {
      //For groups, each node marks its local elements as dependencies
      if constexpr(is_obj<ProxyT>::value){
        auto local_elm = proxy[vt::theContext()->getNode()];
        holder.deps()[holders[local_elm]] = -1;
      } else {
        for(auto index : vt::theCollection()->getLocalIndices(proxy)){
          auto local_elm = proxy[index];
          holder.deps()[holders[local_elm]] = -1;
        }
      }
    }
  }

  template<typename ProxyT>
  void VTContext::add_reg_mapping(size_t hash, ProxyT proxy){
    holders.add_reg_mapping(hash, proxy);
  }

  template<typename ProxyT>
  ProxyHolder& VTContext::get_holder(ProxyT proxy){
    return holders[proxy];
  }

  template<typename ProxyT>
  void VTContext::restart_proxy(ProxyT proxy, ProxyHolder& holder){
    //Restart the holder's metadata for the current version
    constexpr bool IS_GLOBAL = is_elm<ProxyT>::value;
    m_backend->restart(
        holder.label(),
        holder.restarted_version(),
        {holder.metadata_registration(), holder.data_registration()},
        IS_GLOBAL
    );

    if(!holder.tracked()) return;

    bool _check_missing = false;
    if constexpr((not is_elm<ProxyT>::value) and is_col<ProxyT>::value) {
      //Dynamic collections may have elements that we need to reinsert into the
      //collection before we can recover them.
      _check_missing = vt::theCollection()->getDynamicMembership<
                        typename elm_type<ProxyT>::type
                      >(holder.proxy_bits);
    }
    const bool check_missing = _check_missing;
    std::shared_ptr<std::unordered_set<ProxyID>> missing_elms = std::make_shared<std::unordered_set<ProxyID>>();

    vt::EpochType check_missing_epoch;
    if(check_missing) {
      //TODO: How do I know asynchronous collective epoch is matched? Is the label enough?
      check_missing_epoch = vt::theTerm()->makeEpochCollective(
        fmt::format("Detect missing elements of {} @{}", holder, holder.restarted_version())
      );
    }

     
    //Go through dependencies and tell them this minimum required version.
    for(auto dep_iter : holder.deps()){
      ProxyHolder* dep_holder = holders[dep_iter.first];

      if(dep_holder == nullptr) {
        fmt::print(stderr, "WARNING: could not find {} which {} depends upon!\n",
            dep_iter.first, holder);
        continue;
      }

      const bool dep_local = dep_holder->is_local();

      if(!dep_local || (dep_holder->restarted_version() < dep_iter.second)){
        dep_holder->restarted_version(dep_iter.second);
      }

      if(check_missing && 
          dep_holder->proxy_bits == holder.proxy_bits &&
          !dep_local){
        vt::theMsg()->pushEpoch(check_missing_epoch);
        dep_holder->check_missing(missing_elms.get());
        vt::theMsg()->popEpoch(check_missing_epoch);
      }
    }
    
    if(check_missing){
      vt::theTerm()->addAction(check_missing_epoch, [proxy, missing_elms]{
        if constexpr(is_col<ProxyT>::value and not is_elm<ProxyT>::value){
          auto token = proxy.beginModification();
          for(auto& elm : *(missing_elms.get())){
            reindex(proxy, elm.index_bits).insert(token);
          }
          proxy.finishModification(std::move(token));
        }
      });
      vt::theTerm()->finishedEpoch(check_missing_epoch);
    }
    
   
    //Groups just need to update metadata and request versions
    //Elements continue on to restart data
    if(!holder.is_element()) return;

    //A newer version may have been requested by the time
    //all of the updates finish, so register a restart callback
    //but only perform if versions match.
    const int m_version = holder.restarted_version();

    vt::theTerm()->addAction(vt::theTerm()->getEpoch(), [&, this, proxy, m_version] {
      if(m_version == holder.restarted_version() && holder.tracked()){
        //Finally, restart the data of the proxy element
        this->m_backend->restart(
            holder.label(),
            holder.restarted_version(),
            {holder.data_registration()},
            true
        );
      }
    });
  }

  template<typename ProxyT, typename ArgT>
  void VTContext::send_action(
      ProxyT proxy, 
      ProxyAction action,
      const ArgT& arg
  ) {
#ifdef VTCONTEXT_LOG_SENDS
    fmt::print(stderr, "{} sends {} to {}\n", m_proxy, action, proxy);
#endif

    if constexpr(Util::VT::is_elm<ProxyT>::value){
      using ObjT = typename Util::VT::elm_type<ProxyT>::type;
      proxy.template send< 
        &VTContext::remote_action_handler<ProxyT, ObjT, ArgT>
      >(contexts_proxy, proxy, action, arg);
    } else {
      contexts_proxy.template broadcast<
        &VTContext::remote_action_handler<ProxyT, ArgT>
      >(proxy, action, arg);
    }
  }
  
  template<typename ProxyT, typename ArgT>
  void VTContext::send_action(
      int dest,
      ProxyT proxy, 
      ProxyAction action,
      const ArgT& arg
  ) {
#ifdef VTCONTEXT_LOG_SENDS
    fmt::print(stderr, "{} sends {} to {}\n", m_proxy, action, contexts_proxy[dest]);
#endif

    contexts_proxy[dest].template send<
      &VTContext::remote_action_handler<ProxyT, ArgT>
    >(proxy, action, arg);
  }
    
  
  template<typename ProxyT, typename ObjT, typename ArgT>
  void VTContext::remote_action_handler(
      ObjT* unused,
      VTContextProxy ctx_proxy,
      ProxyT proxy,
      ProxyAction action,
      ArgT arg
  ) {
    VTContext* ctx = vt::theObjGroup()->get(ctx_proxy);
#ifdef VTCONTEXT_LOG_RECEIVES
    fmt::print(stderr, "{} recvs {} for {}\n", ctx->m_proxy, action, proxy);
#endif

    constexpr bool REMOTE_REQUEST = true;
    ctx->action_handler(proxy, ctx->get_holder(proxy), action, arg, REMOTE_REQUEST);
  }

  template<typename ProxyT, typename ArgT>
  void VTContext::remote_action_handler(
      ProxyT proxy,
      ProxyAction action,
      ArgT arg
  ) {
#ifdef VTCONTEXT_LOG_RECEIVES
    fmt::print(stderr, "{} recvs {} for {}\n", m_proxy, action, proxy);
#endif

    constexpr bool REMOTE_REQUEST = true;
    action_handler(proxy, get_holder(proxy), action, arg, REMOTE_REQUEST);
  }

  struct ProxyMigrateInfo {
    ProxyStatus status;
    bool modified;
    std::vector<std::string> registered_regions;
    int new_owner;

    template<typename SerT>
    void serialize(SerT& s){
      s | status | modified | registered_regions;
    }
  };

  template<typename ProxyT>
  std::any VTContext::action_handler(
      ProxyT proxy,
      ProxyHolder& holder,
      ProxyAction action,
      std::any arg,
      bool remote_request
  ) {
    constexpr bool elm = is_elm<ProxyT>::value;
    constexpr bool group = !elm;
    constexpr bool col = is_col<ProxyT>::value;
    constexpr bool obj = is_obj<ProxyT>::value;

    const bool local = is_local(proxy);

    //Group actions should be broadcasted to all contexts
    const bool need_broadcast = group && !remote_request;
    //Element actions should be sent to the local context
    const bool need_unicast   = elm && !local;

    //Other than exclusively-local operations
    const bool need_send = need_broadcast || need_unicast;

#ifdef VTCONTEXT_LOG_EVENTS
    fmt::print("{} processing {} on {}. Remote: {}, Broadcast: {}, Unicast: {}\n", 
                m_proxy, action, proxy, remote_request, need_broadcast, need_unicast);
#endif

    switch(action){
      case GET_HOLDER_AT:
        return &holders[reindex(proxy, std::any_cast<uint64_t>(arg))];

      case FETCH_STATUS:
        if(!local) {
          assert(!remote_request);
          //Fetch from the proxy's local context
          send_action(proxy, action, int(vt::theContext()->getNode()));
        } else if(remote_request) {
          //Reply to remote requester after checking that my ready epoch is updated
          send_action(std::any_cast<int>(arg), proxy, SET_STATUS, holder.get_status());
        } else {
          assert(false);
        }
        return nullptr;

      case SET_STATUS:
        assert(remote_request);
        //Might be local if fetch_status interrupted by the element migrating here
        if(!local) holder.set_status(std::any_cast<ProxyStatus>(arg));
        return nullptr;

      case CHECK_LOCAL:
        assert(!remote_request);
        return local;

      case SET_TRACKED:
        if(need_unicast) 
          send_action(proxy, action, std::any_cast<bool>(arg));
        else if(need_broadcast && holder.tracked() != std::any_cast<bool>(arg))
          send_action(proxy, action, std::any_cast<bool>(arg));
        else if(remote_request) 
          holder.tracked(std::any_cast<bool>(arg));
        return nullptr;

      case SET_CHECKPOINTED_VERSION:
        if(need_unicast) send_action(proxy, action, std::any_cast<int>(arg));
        else if(remote_request) holder.checkpointed_version(std::any_cast<int>(arg));
        return nullptr;

      case SET_RESTARTED_VERSION:
        if(need_send) {
          send_action(proxy, action, std::any_cast<int>(arg));
        } else {
          int requested_version = std::any_cast<int>(arg);
          bool should_update = requested_version > holder.restarted_version();
          if(should_update) {
            holder._status.restarted_version = requested_version;
            if(need_broadcast){
              send_action(proxy,action, requested_version);
            }

            assert(!elm || local);
            restart_proxy(proxy, holder);
          }
        }
        return nullptr;

      case MODIFY:
        if(need_unicast){
          send_action(proxy, action, nullptr);
        } else if(need_broadcast){
          if(modified_proxies.find(proxy) == modified_proxies.end()){
            send_action(proxy, action, nullptr);
          }
        } else if(modified_proxies.insert(proxy).second) {
          holder.checkpointed_version(holder.checkpointed_version()+1);
          if constexpr(not elm) {
            if constexpr(obj) {
              auto& child_holder = holders[proxy[vt::theContext()->getNode()]];
              if(modified_proxies.insert(child_holder).second)
                child_holder.checkpointed_version(child_holder.checkpointed_version()+1);
            } else {
              for(auto index : vt::theCollection()->getLocalIndices(proxy)){
                auto& child_holder = holders[proxy[index]];
                if(modified_proxies.insert(child_holder).second)
                  child_holder.checkpointed_version(child_holder.checkpointed_version()+1);
              }
            }
          }
        }
        return nullptr;

      case REGISTER:
        if(need_unicast){
          send_action(proxy, action, std::any_cast<std::string>(arg));
        } else { 
          if(remote_request) {
            assert(elm);
            //Register as a member to ContextBase
            this->register_to(std::any_cast<std::string>(arg), proxy, "");
          }
          holder.tracked(true);
          holder.modified();
        }
        return nullptr;

      case DEREGISTER:
        if(need_send){
          send_action(proxy, action, std::any_cast<std::string>(arg));
        } else { 
          if(remote_request) {
            //Deregister as a member to ContextBase
            this->deregister_from(std::any_cast<std::string>(arg), proxy, "");
          }
          holder.tracked(false);
          holder.modified();
        }
        return nullptr;

      case CHECK_DYNAMIC:
        if constexpr(col){
          return vt::theCollection()->getDynamicMembership<
                    typename elm_type<ProxyT>::type
                 >(holder.proxy_bits);
        }
        return bool(false);

      case CHECK_MISSING:
        if constexpr(elm && col) {
          if(!local){
            vt::theCollection()->getElementLocation(
                proxy,
                [proxy, arg](vt::NodeType location){
                  if(location == vt::uninitialized_destination){
                    std::any_cast<std::unordered_set<ProxyID>*>(arg)->insert(proxy);
                  }
                },
                false
            );
          }
        }
        return nullptr;

      case DEREGISTER_EVENT_LISTENER:
        if constexpr(col && !elm) {
          using ObjT = typename elm_type<ProxyT>::type;
          vt::theCollection()->unregisterElementListener<ObjT>(
              holder.proxy_bits, std::any_cast<int>(arg)
          );
        } else {
          assert(false);
        }
        return nullptr;

      case MIGRATE_STATUS:
        assert(col and elm);
        if(!local){
          assert(!remote_request);

          ProxyMigrateInfo info;
          info.status = std::any_cast<ProxyStatus>(arg);
          info.modified = modified_proxies.erase(proxy);
          
          //Deregister from any ContextBase regions and inform new host of them.
          Registration core_reg = create_registration<ProxyT>(*this, proxy);
          for(auto& region_pair : regions){
            if(region_pair.second.erase(core_reg)){
              info.registered_regions.push_back(region_pair.first);
            }
          }
          if(!info.registered_regions.empty()) m_backend->deregister_member(core_reg);
          
          send_action(proxy, MIGRATE_STATUS, info);
        } else {
          assert(remote_request);
          ProxyMigrateInfo info = std::any_cast<ProxyMigrateInfo>(arg);
          holder.migrated_status(info.status);
          if(info.modified) modified_proxies.insert(proxy);

          Registration core_reg = create_registration<ProxyT>(*this, proxy);
          for(std::string& region_label : info.registered_regions){
            regions[region_label].insert(core_reg);
          }
          if(!info.registered_regions.empty()) m_backend->register_member(core_reg);
        }
        return nullptr;
    }
    assert(false);
    return nullptr;
  }
}


//Define how fmt should format some of the things used in logging

namespace fmt {
template<>
struct formatter<KokkosResilience::Context::VT::ProxyAction> 
    : formatter<string_view> 
{
  using Action = KokkosResilience::Context::VT::ProxyAction;
  auto format(const Action& action, format_context& ctx){
    constexpr std::array names = {
      KR_VT_PROXY_ACTIONS(KR_VT_ENUM_LIST_STR)
    };
    return fmt::format_to(ctx.out(), "{}", names[action]);
  }
};

template<>
struct formatter<::vt::vrt::collection::listener::ElementEventEnum> 
    : formatter<string_view> 
{
  using Event = ::vt::vrt::collection::listener::ElementEventEnum;
  
  static constexpr
  std::string_view to_string(const Event& event){
    switch(event) {
      case Event::ElementCreated: return "ElementCreated";
      case Event::ElementDestroyed: return "ElementDestroyed";
      case Event::ElementMigratedOut: return "ElementMigratedOut";
      case Event::ElementMigratedIn: return "ElementMigratedIn";
    }
    return "Unknown vt::vrt::collection::listener::ElementEventEnum";
  }

  auto format(const Event& event, format_context& ctx){
    return fmt::format_to(ctx.out(), "{}", to_string(event));
  }
};

template<typename ProxyT>
struct formatter<ProxyT, typename KokkosResilience::Util::VT::is_proxy<ProxyT, char>::type>
    : formatter<string_view> 
{
  auto format(ProxyT proxy, format_context& ctx){
    return fmt::format_to(ctx.out(), "{}", KokkosResilience::Util::VT::proxy_label(proxy));
  }
};

template<>
struct formatter<KokkosResilience::Context::VT::ProxyHolder>
    : formatter<string_view> 
{
  using Holder = KokkosResilience::Context::VT::ProxyHolder;
  auto format(const Holder& holder, format_context& ctx){
    return fmt::format_to(ctx.out(), "{}", holder.label());
  }
};

template<>
struct formatter<KokkosResilience::Util::VT::ProxyID>
    : formatter<string_view> 
{
  using ProxyID = KokkosResilience::Util::VT::ProxyID;
  auto format(const ProxyID& proxy_id, format_context& ctx){
    if(proxy_id.is_element()){
      return fmt::format_to(ctx.out(), "ProxyElement({}[{}])", proxy_id.proxy_bits, proxy_id.index_bits);
    } else {
      return fmt::format_to(ctx.out(), "Proxy({})", proxy_id.proxy_bits);
    }
  }
};
}

#endif
